# Experiment repository

- This directory alone is the Git/VS Code project root. Do not upload the parent research workspace.
- Default deployment: develop locally in VS Code and explicitly upload changed project files using its SFTP extension to `/ceph_data/leetj3610/experiment`. GitHub push followed by Seraph clone/pull is an alternative when not using VS Code. Do not use ZIP deployment or competing synchronization methods.
- Configure SFTP exclusions separately from .gitignore: exclude .git, .vscode, credentials, environments, vendor, datasets, models, caches, logs and outputs. Preserve server-only files; do not enable remote deletion or automatic upload-on-save by default. Keep shell scripts LF-terminated.
- Never commit credentials, connection settings, datasets, model weights, caches, upstream vendor checkouts, logs or run outputs.
- On Seraph, keep the checkout under `/ceph_data/leetj3610/experiment`; keep model/cache/output files on Ceph, Conda under `/data/leetj3610/anaconda3`, and datasets in the allocated node's permitted local dataset directory.
- Run installation, Python, data preparation and inference only in an allocated compute job. No VS Code Remote SSH.
- Keep upstream SD-square at the revision in configs/smoke.json and unmodified. Do not change pins, tolerances or research scope to conceal failures.
- Do not upload, pull or edit files used by running or queued experiments. Update between jobs. When switching from SFTP to Git, inspect local and remote changes before `git pull --ff-only`; never overwrite or reset them automatically. Git HEAD alone does not identify SFTP-uploaded code.
- Default new sbatch jobs to 24 hours (`--time=1-00:00:00`) on batch_ugrad, subject to current policy/account limits. This is the user's preference, not a school requirement. Do not shorten it without an explicit reason and user agreement. Short interactive debug allocations remain separate; never change existing jobs implicitly.
- CPU tests are not real-checkpoint GPU results. Report actual tests and pending server validation separately.
- In the full research workspace, read ../research_core.md before scientific design decisions and ../SERAPH_POLICY_GUIDE.md before Seraph work. If those documents are unavailable in a standalone clone, request the current policy/specification rather than inventing it.
- Give Korean operational instructions one stage at a time, including where to run them, their purpose and expected output.
- User standing request (2026-09-18): when server logs arrive, continue autonomously through diagnosis, in-scope local fixes, tests and concrete upload/run instructions; do not stop at a proposal or ask repeatedly to implement. Continue until the user revokes this preference. User still performs SFTP and server execution; this does not authorize remote authentication, tolerance relaxation, or unapproved research-scope changes. Request related failure evidence together.
