# Agent environment artifact setup (Codex Cloud + Claude Code Web)

This repo ships setup scripts that download a prebuilt conda environment artifact from GitHub Actions and unpack it in the web agent container.

- `agent_scripts/codex_setup_container.sh` → for **Codex Cloud**
- `agent_scripts/claude_code_web_setup_container.sh` → for **Claude Code Web**

The artifact is built by `.github/workflows/build-agent-env-artifact.yml` on pushes to `main` (and can also be run manually with **workflow_dispatch**).

## Network access requirements

These setup scripts require outbound internet access. They will not work in a fully offline or no-egress container.

### Minimum allow-list for artifact download

Using the current setup script flow (`list artifacts` + `download artifact zip`), the required domains are:

- `api.github.com` (GitHub Actions artifacts API)
- `*.blob.core.windows.net` (signed Azure Blob URL that GitHub redirects to for artifact bytes)

In a live query against this repo, the redirect resolved to `productionresultssa14.blob.core.windows.net` (the exact subdomain can vary), so allow-list the wildcard form.

### Additional domains commonly needed by the full setup scripts

Both setup scripts also run package install steps, so most environments should additionally allow:

- Ubuntu/Debian apt mirrors (for `apt -y install pandoc`)
- Python package index endpoints (for `python -m pip install -e '.[dev]'`, typically `pypi.org` and `files.pythonhosted.org`)

## 1) Create a GitHub token for artifact download

You need a token because the setup scripts call the GitHub Actions Artifacts API.

### Recommended: Fine-grained personal access token

1. In GitHub, open **Settings → Developer settings → Personal access tokens → Fine-grained tokens**.
2. Click **Generate new token**.
3. Select the `lincc-frameworks/hyrax` repository.
4. Grant these repository permissions:
   - **Actions: Read**
   - **Contents: Read**
5. Set an expiration date and create the token.
6. Copy it once and store it securely.

> You can also use a classic PAT with `repo` scope if your org policy requires it, but fine-grained + least privilege is preferred.

## 2) Ensure the artifact exists

Before provisioning a web environment, make sure the artifact has been built at least once from `main`:

1. Go to **Actions → Build agent conda environment artifact**.
2. Confirm a successful run on `main`.
3. Confirm artifact `hyrax-agent-conda-env-main` exists.

## 3) Provision on Codex Cloud

In your Codex Cloud web environment configuration:

1. Add environment variable:
   - `GITHUB_TOKEN=<your token>`
2. Set setup script to:

```bash
./agent_scripts/codex_setup_container.sh
```

What happens:
- Installs `pandoc`
- Downloads `hyrax-agent-conda-env-main`
- Unpacks to `$HOME/hyrax-venv`
- Runs `conda-unpack`
- Runs `pip install -e '.[dev]'` in this repo

## 4) Provision on Claude Code Web

In your Claude Code Web environment:

1. Add environment variable:
   - `GITHUB_TOKEN=<your token>`
2. Use this setup script:

```bash
#!/bin/bash
./hyrax/agent_scripts/claude_code_web_setup_container.sh
```

This script performs the same environment restoration and editable install as Codex Cloud.

## Troubleshooting

- **401/403 from GitHub API**: token is missing/invalid, expired, or lacks `Actions: Read`.
- **No artifact found**: workflow has not run successfully on `main` yet.
- **`conda-unpack` not found**: artifact may be stale/corrupt; re-run workflow on `main`.
