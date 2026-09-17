# Experiment repository

- This directory alone is the Git/VS Code project root. Do not upload the parent research workspace.
- Develop locally, commit and push to GitHub, then clone/pull on Seraph. Do not use ZIP deployment or simultaneous SFTP code synchronization.
- Never commit credentials, connection settings, datasets, model weights, caches, upstream vendor checkouts, logs or run outputs.
- On Seraph, keep the checkout under `/ceph_data/leetj3610/experiment`; keep model/cache/output files on Ceph, Conda under `/data/leetj3610/anaconda3`, and datasets in the allocated node's permitted local dataset directory.
- Run installation, Python, data preparation and inference only in an allocated compute job. No VS Code Remote SSH.
- Keep upstream SD-square at the revision in configs/smoke.json and unmodified. Do not change pins, tolerances or research scope to conceal failures.
- Stop active experiment processes before pulling code. Use `git pull --ff-only`; inspect local changes instead of overwriting or resetting them.
- CPU tests are not real-checkpoint GPU results. Report actual tests and pending server validation separately.
- In the full research workspace, read ../research_core.md before scientific design decisions and ../SERAPH_POLICY_GUIDE.md before Seraph work. If those documents are unavailable in a standalone clone, request the current policy/specification rather than inventing it.
- Give Korean operational instructions one stage at a time, including where to run them, their purpose and expected output.
