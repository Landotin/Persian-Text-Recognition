# Project context and working rules

- Start with `PROGRESS.md`. Read only relevant `PROJECT_PLAN.md` sections; open `PHASE_1_REPORT.md` for audit evidence. Avoid rereading the conversation or loading large manifests unless needed.
- Update `PROGRESS.md` after each milestone with actual work, decisions, pending items, and the next action. Update `PROJECT_PLAN.md` when scope or architecture changes.
- Keep DenseNet121 and the stable 33-label order. Handwritten آ remains unvalidated until reviewed manual samples are integrated.
- Phase 1 audit and Phase 2 notebook preparation are complete. The seed-42 Colab prototype and its verified model bundle exist. The frozen test split received one post-hoc diagnostic after checkpoint selection; do not tune this checkpoint using those results. Any redesign informed by the diagnostic needs a fresh independent holdout.
- Preserve frozen archive source-form and font-group splits. Keep independent Persian-reader review pending unless a reader actually performs it. Validation demos and one-time prototype diagnostics are not the planned multi-seed, writer-disjoint final study.
- Do not start another Colab training run unless the user asks to proceed. Do not claim a model, prediction, or metric without evidence in `PROGRESS.md` or the detailed report.
- The Streamlit inference app has been launched locally and smoke-checked with a saved validation glyph. Do not present that single example as a performance estimate.
- Use Markdown files as durable context across compaction. A checkpoint is not proof of deployment.
