# Spec: GitHub Copilot Cloud Agent Environment Setup

## Background

Hyrax already ships pre-built conda environment artifacts for two AI coding agents:

| Agent | Setup mechanism |
|-------|----------------|
| **Codex Cloud** | `agent_scripts/codex_setup_container.sh` (called from the Codex web config) |
| **Claude Code Web** | `agent_scripts/claude_code_web_setup_container.sh` + `.claude/hooks/session-start.sh` |

Both scripts download the `hyrax-agent-conda-env-main` artifact built by
`.github/workflows/build-agent-env-artifact.yml`, unpack it with `conda-unpack`,
and install hyrax in editable mode.

This spec describes the equivalent setup for **GitHub Copilot cloud agent**, which
runs in a GitHub Actions environment and supports a dedicated setup workflow at
`.github/workflows/copilot-setup-steps.yml`.

## Goals

1. Give GitHub Copilot cloud agent a ready-to-use Hyrax development environment.
2. Reuse the existing `hyrax-agent-conda-env-main` artifact (same build, same deps).
3. Minimise agent startup time by relying on the pre-built artifact rather than
   installing dependencies from scratch on every run.
4. Ensure the environment survives Copilot's bash tool calls (i.e. `python` and
   project tools are on `PATH` in every shell the agent opens).

## Non-goals

* Creating a new artifact-build workflow (reuse the existing one).
* Supporting non-Ubuntu runners (Copilot only supports Ubuntu x64 and Windows x64;
  this implementation targets Ubuntu).

## Design

### Workflow file: `.github/workflows/copilot-setup-steps.yml`

GitHub Copilot cloud agent looks for a job named `copilot-setup-steps` in this
specific workflow file.  The job runs before Copilot starts, and the resulting
runner filesystem is handed off to the agent.

**Steps:**

1. **Checkout code** – needed for the editable `pip install -e '.[dev]'`.
2. **Install system packages** – `pandoc` (required by Sphinx / nbsphinx doc builds).
3. **Download & unpack conda environment** – mirrors `claude_code_web_setup_container.sh`:
   - Fetches the artifact ID from the GitHub Actions API using `GITHUB_TOKEN`
     (`actions: read` permission is sufficient).
   - Follows the GitHub→Azure redirect *without* forwarding the Bearer token
     (Azure rejects it with HTTP 503).
   - Unpacks the `.tar.gz` and runs `conda-unpack` to fix baked-in paths.
   - Touches `.setup-complete` sentinel (consistent with other scripts).
4. **Editable install** – `pip install -e '.[dev]'` from the checked-out repo.
5. **Persist activation in shell profiles** – appends a guarded activation block to
   `~/.bashrc` and `~/.profile` (same pattern as `codex_maintain_container.sh`)
   so every shell Copilot opens automatically activates the hyrax venv.
   Also writes `$HOME/hyrax-venv/bin` to `$GITHUB_PATH` for the remaining steps
   in the setup job itself.

### Permissions

```yaml
permissions:
  contents: read   # for actions/checkout
  actions: read    # for artifact download via GitHub API
```

### Triggers

The workflow runs on:
- `workflow_dispatch` – manual validation from the Actions tab.
- `push` / `pull_request` (path-filtered to the workflow file itself) – automatic
  validation whenever the setup steps are modified.

## Differences from other agent setups

| Aspect | Codex Cloud | Claude Code Web | Copilot cloud agent |
|--------|-------------|-----------------|---------------------|
| Trigger | Codex web config | Claude environment config | `copilot-setup-steps.yml` |
| Shell profile | `.profile` | n/a (session-start hook) | `.bashrc` + `.profile` |
| Editable install | `codex_maintain_container.sh` | `session-start.sh` | `copilot-setup-steps.yml` step |
| Token source | `$GITHUB_TOKEN` env var (user-supplied) | `$GITHUB_TOKEN` env var (user-supplied) | `secrets.GITHUB_TOKEN` (automatic) |
| Azure redirect workaround | No (older script) | Yes | Yes |

## Verification

The workflow can be verified by:
1. Pushing a change to `.github/workflows/copilot-setup-steps.yml` and confirming
   the CI check passes in the pull request.
2. Running the workflow manually from the **Actions → Copilot Setup Steps** tab.
3. Assigning a task to GitHub Copilot and confirming it can run `python -m pytest`
   without installing anything itself.
