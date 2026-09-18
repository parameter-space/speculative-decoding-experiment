# Speculative decoding experiment

SD² S1 diagnostic harness. Read [한국어 실행 안내](README_KO.md) and [compatibility and provenance](COMPATIBILITY.md).

For an external review, start with [코드 검토 안내](docs/REVIEW_MAP_KO.md). Raw run evidence is shared separately, not committed here. The completed smoke uses an explicit FP64 reference policy; the default low-precision path is not a completed reproduction.

Develop only in this repository. Default deployment is local VS Code plus explicit SFTP upload of changed code to `/ceph_data/leetj3610/experiment`; GitHub push and Seraph clone/pull remain an alternative when not using VS Code. Do not sync the parent research folder or use VS Code Remote SSH. New batch jobs default to 24 hours on `batch_ugrad`.

Tests use tiny random CPU models and are not evidence of real-checkpoint GPU compatibility or scientific effects. Model weights, datasets, environments, credentials and run outputs are not distributed in this repository.
