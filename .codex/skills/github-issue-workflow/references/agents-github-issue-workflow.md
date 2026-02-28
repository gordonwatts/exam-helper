# AGENTS.md Extract: GitHub Issue Workflow

Source: repository `AGENTS.md`

## GitHub Issue Workflow

- Use `gh` for issue and PR status in this repo when available.
- List open issues with:
  - `gh issue list --state open --limit 200`
- If `gh` is not installed or not available on PATH, ask the user to install/authenticate `gh`, then retry the same command.

## Standard Issue Development Flow

1. Check out `main` and update it before starting issue work.
   - Always perform this update on `main` before any issue-branch pull/sync work.
   - If sandbox/network restrictions block the update command, request an elevated command and retry.
2. Create a dedicated branch for the issue (for example, `issue-123-short-name`) before making any code or docs edits.
3. Implement and test changes on that branch (never do issue implementation work directly on `main`).
4. Open a draft PR from the branch into `main` as soon as implementation starts.
5. Keep the PR in draft until tests and review items are complete, then mark it ready for review.
6. In the PR description, include an issue-closing reference such as `fix #123` so GitHub automatically closes the issue when the PR is merged.
7. Run local tests before every push that updates an issue branch/PR (at minimum `uv run --extra dev pytest -q` unless a narrower test scope is explicitly justified in the PR).
