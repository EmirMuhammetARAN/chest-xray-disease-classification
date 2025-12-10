# Contributing

Thanks for your interest in improving this project! This document explains how to contribute in a way that keeps the repository clean and reviewable.

## How to contribute

- Fork the repository and create a branch: `git checkout -b feature/your-feature`
- Work locally and keep commits small and focused
- Rebase or merge the main branch before opening a PR
- Open a pull request with a clear title and description of changes

## Code style

- Use Python 3.8+.
- Follow PEP8. Prefer `black` for formatting and `isort` for imports.
- Example (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install black isort
black .
isort .
```

## Tests

- Add unit tests for new functionality under the `test/` directory.
- Keep tests deterministic and small.

## Model/data files

- Large model checkpoints are committed (if present). Do not re-upload large checkpoints unless necessary.
- Do not commit raw NIH ChestX-ray14 data to the repo. Add scripts or pointers to download the dataset instead.

## PR checklist

- [ ] My PR has a clear title and description
- [ ] I ran formatting (`black`) and import sorting (`isort`)
- [ ] I added or updated tests if applicable
- [ ] I updated `README.md` or other docs if behavior changed

## Review process

- PRs will be reviewed by the maintainer. Expect feedback on clarity and reproducibility.
- Maintain backward compatibility where reasonable; document breaking changes.

## Contact

If you need guidance before starting, open an issue describing your idea.
