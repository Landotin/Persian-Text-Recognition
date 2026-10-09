"""Exercise the saved-snapshot notebook without audit, fonts, model, or test images."""
import contextlib
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebooks/01_data_preparation_training.ipynb'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class NotebookReuseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.snapshot = self.root / 'snapshot'
        (self.snapshot / 'audit').mkdir(parents=True)
        (self.root / 'config').mkdir()
        (self.root / 'notebooks').mkdir()
        self.nb = json.loads(NOTEBOOK.read_text())
        (self.root / 'notebooks' / NOTEBOOK.name).write_text(NOTEBOOK.read_text())
        labels = json.loads((ROOT / 'config/characters.json').read_text())
        (self.root / 'config/characters.json').write_text(json.dumps(labels))
        code = ''.join(self.nb['cells'][9]['source'])
        preparation_source = code[code.index('def prepare_image('):code.index('\ndef show_preprocessing_examples')].rstrip() + '\n'
        settings_node = next(node.value for node in ast.parse(code).body
                             if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'PREPROCESSING' for t in node.targets))
        preprocessing = eval(compile(ast.Expression(settings_node), '<settings>', 'eval'),
                             {'OUTPUT_SIZE': 224, 'CONTENT_SIZE': 192})
        preprocessing['source_sha256'] = hashlib.sha256(preparation_source.encode()).hexdigest()
        (self.snapshot / 'preprocessing_source.py').write_text(preparation_source)
        (self.snapshot / 'preprocessing.json').write_text(json.dumps(preprocessing))
        font_code = ''.join(self.nb['cells'][11]['source'])
        groups_node = next(node.value for node in ast.parse(font_code).body
                          if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'FONT_GROUP_SPLITS' for t in node.targets))
        frozen_groups = ast.literal_eval(groups_node)
        groups = {'family_to_group': {'train': 'gandom', 'val': 'noto_sans_arabic', 'test': 'noto_kufi_arabic'},
                  'family_split': {'train': 'train', 'val': 'val', 'test': 'test'}, 'group_split': frozen_groups}
        (self.snapshot / 'font_splits.json').write_text(json.dumps(groups))
        checks = {'checks': [{'blank_or_missing_characters': [], 'alef_maddah_difference_pixels': 1}]}
        (self.snapshot / 'font_render_check.json').write_text(json.dumps(checks))
        self.archive = self.root / 'raw.zip'
        rows, frozen = [], []
        self.raw_files = []
        with zipfile.ZipFile(self.archive, 'w') as zf:
            for split in ['train', 'val', 'test']:
                for target, character in enumerate(labels['class_order'][:32]):
                    member = f'{split}/{target}/sample.png'
                    if split != 'test':
                        image = Image.new('L', (56, 56), 0)
                        image.putpixel((20, 20), 255)
                        data = io.BytesIO(); image.save(data, 'PNG')
                        zf.writestr(member, data.getvalue())
                    group = {'train': 10, 'val': 110, 'test': 109}[split]
                    frozen.append({'archive_member': member, 'split': split, 'source_group': group})
                    rows.append({'archive_member': member, 'image_path': '', 'split': split,
                        'character': character, 'target_idx': target, 'source_group': group,
                        'source_type': 'handwritten', 'is_prepared': False, 'writer_id': ''})
                for target, character in enumerate(labels['class_order']):
                    image_path = self.root / f'printed_{split}_{target}.png'
                    if split != 'test':
                        image = Image.new('L', (224, 224), 255)
                        image.putpixel((100, 100), 0); image.save(image_path)
                        self.raw_files.append(image_path)
                    rows.append({'archive_member': '', 'image_path': str(image_path), 'split': split,
                        'character': character, 'target_idx': target, 'source_group': groups['family_to_group'][split],
                        'source_type': 'printed', 'is_prepared': True, 'writer_id': ''})
        self.archive_sha = sha(self.archive)
        pd.DataFrame(frozen).to_csv(self.snapshot / 'audit/letter_manifest.csv', index=False)
        (self.snapshot / 'audit/audit.json').write_text(json.dumps({'archive_sha256': self.archive_sha, 'decode_failures': []}))
        pd.DataFrame(rows).to_csv(self.snapshot / 'training_manifest.csv', index=False)
        config = {'archive_sha256': self.archive_sha, 'class_order': labels['class_order'],
                  'label_version': labels['label_version'], 'seed': 42, 'preprocessing': preprocessing,
                  'font_groups': groups, 'manifest_sha256': {'training_manifest.csv': sha(self.snapshot / 'training_manifest.csv')},
                  'font_render_check_sha256': sha(self.snapshot / 'font_render_check.json'),
                  'reader_review': 'pending', 'final_test_used': False}
        (self.snapshot / 'run_config.json').write_text(json.dumps(config))
        self.env = {'PERSIAN_PROJECT_DIR': str(self.root), 'PERSIAN_PREPARATION_SOURCE_DIR': str(self.snapshot),
                    'PERSIAN_ARCHIVE_PATH': str(self.archive), 'PERSIAN_ARTIFACT_ROOT': str(self.root / 'artifacts'),
                    'PERSIAN_RUNTIME_CACHE_DIR': str(self.root / 'runtime')}

    def run_notebook(self, forbid_raw=False):
        namespace = {}
        with patch.dict(os.environ, self.env), contextlib.redirect_stdout(io.StringIO()):
            for i, cell in enumerate(self.nb['cells']):
                if cell['cell_type'] != 'code':
                    continue
                code = ''.join(cell['source'])
                exec(compile(code, f'notebook-cell-{i}', 'exec'), namespace)
                if i == 1:
                    namespace['EXPECTED_ARCHIVE_SHA256'] = self.archive_sha
                    namespace['EXPECTED_FROZEN_MANIFEST_SHA256'] = sha(self.snapshot / 'audit/letter_manifest.csv')
                    namespace['LOCAL_ARCHIVE'] = self.archive
                    if forbid_raw:
                        def forbidden():
                            raise AssertionError('A cache hit opened the raw archive.')
                        namespace['ensure_local_archive'] = forbidden
                elif i == 2:
                    namespace['display'] = lambda *_: None
        return namespace

    def test_saved_preparation_then_source_free_cache_hit(self):
        first = self.run_notebook()
        self.assertFalse(first['cache_info']['cache_hit'])
        self.assertEqual(len(first['cached_prototype_manifest']), 130)
        self.assertEqual(set(first['cached_prototype_manifest']['split']), {'train', 'val'})
        self.assertTrue(first['PROTOTYPE_READY'])
        self.assertFalse(first['FINAL_READY'])
        self.assertEqual(first['PREPARED_RUN_ID'], first['RUN_ID'])
        for name in ['run_config.json', 'preprocessing.json', 'preprocessing_source.py',
                     'font_splits.json', 'font_render_check.json', 'prepared_cache_config.json']:
            self.assertTrue((first['RUN_DIR'] / name).is_file())
        self.archive.unlink()
        for file in self.raw_files:
            file.unlink()
        # A fresh runtime must recover images from the durable package alone.
        import shutil
        shutil.rmtree(self.root / 'runtime')
        second = self.run_notebook(forbid_raw=True)
        self.assertTrue(second['cache_info']['cache_hit'])
        self.assertEqual(first['cache_info']['cache_id'], second['cache_info']['cache_id'])
        frame = second['cached_prototype_manifest']
        self.assertTrue(all(Path(p).is_file() for p in frame['prepared_image_path']))
        self.assertTrue(all(Path(p).is_file() for p in frame.loc[frame['split'] == 'val', 'raw_preview_image_path']))

    def test_changed_executing_preparation_is_rejected(self):
        original = self.nb['cells'][9]['source']
        self.nb['cells'][9]['source'] = ''.join(original).replace('border)) < 127.5', 'border)) < 126.5').splitlines(keepends=True)
        with self.assertRaisesRegex(RuntimeError, 'executing preparation differs'):
            self.run_notebook()

    def test_cached_loader_and_previews_have_no_raw_preparation_calls(self):
        model_cell = ''.join(self.nb['cells'][19]['source'])
        dataset_code = model_cell[model_cell.index('    class CharacterDataset'):model_cell.index('    def build_loaders')]
        preview_code = model_cell[model_cell.index('    def _read_raw_preview_image'):model_cell.index('    def export_validation_predictions')]
        for forbidden in ['prepare_image(', 'ZipFile(', 'ensure_local_archive(']:
            self.assertNotIn(forbidden, dataset_code + preview_code)


if __name__ == '__main__':
    unittest.main()
