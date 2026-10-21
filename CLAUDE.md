# Notes for Claude sessions

- Project: roof snow-clearance checker for a Moscow client. Language of user-facing docs: Russian; code and comments: English.
- Read `docs/research-notes.md` first: it records verified facts, licence decisions and the rework queue.
- Licences: commercial use only. Never add SuperPoint, DSINE, SegFormer, Ultralytics YOLO, LightlyTrain, AlbumentationsX,
  TruFor/Noiseprint, DA-V2 Base/Large or DA3 Large/Giant. Fab/Megascans NoAI assets never go into generative models.
- Data, weights and outputs are not committed (see `.gitignore`). Client photos stay out of git.
- Checks: `python3 tests/stage1.py` (registration/overlays/rectification), `python3 tests/synth_batch.py 10` + `tests/stage2_sheet.py`.
