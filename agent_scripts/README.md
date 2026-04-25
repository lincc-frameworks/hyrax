# Agent environment artifact setup (Codex Cloud + Claude Code Web)

This repo ships setup scripts that download a prebuilt conda environment artifact from GitHub Actions and unpack it in the web agent container.

- `agent_scripts/codex_setup_container.sh` → for **Codex Cloud**
- `agent_scripts/claude_code_web_setup_container.sh` → for **Claude Code Web**

The artifact is built by `.github/workflows/build-agent-env-artifact.yml` on pushes to `main` (and can also be run manually with **workflow_dispatch**).

## Network access requirements

These setup scripts require outbound internet access. They will not work in a fully offline or no-egress 
container. Right now you need to set full internet access to use this. See the end of the file for known
information about custom allow-lists.

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

In your Codex Cloud web [environment](https://chatgpt.com/codex/cloud/settings/environments) configuration:

1. Set Python version to 3.11-3.13 in preinstalled packages.
2. Add environment variable:
   - `GITHUB_TOKEN=<your token>`
3. Enable container caching
4. Set setup script to:

```bash
/workspace/hyrax/agent_scripts/codex_setup_container.sh
```
5. Set maintenance script to:

```bash
/workspace/hyrax/agent_scripts/codex_maintain_container.sh
```
6. Turn on agent internet access, with the "All (unrestricted)" allowlist and all HTTP methods

What happens:
- Installs `pandoc`
- Downloads `hyrax-agent-conda-env-main`
- Unpacks to `$HOME/hyrax-venv`
- Runs `conda-unpack`
- Runs `pip install -e '.[dev]'` in this repo

## 4) Provision on Claude Code Web

From the new session start page in claude code on the web, click the environment name and then the
gear icon next to the environment you would like to enable.

1. Enable Full internet access
2. Use this setup script:

```bash
#!/bin/bash
# Cache bust v1
export GITHUB_TOKEN=<your token>
./hyrax/agent_scripts/claude_code_web_setup_container.sh
```

This setup script restores the prebuilt environment artifact in the container, similar to Codex Cloud.
The editable install is not performed during container setup; it runs later from
`.claude/hooks/session-start.sh` when the Claude Code Web session starts.
Claude code has no cache switch, so changing the cache buster comment line can force re-runs of the setup 
script if your claude code environment shows signs of being stale.

## Custom allow-lists

The list outlined below is known to be insufficient; however, it is provided in the hope that 
this can be made to work in the future.

Use the provider defaults for package managers to allow package installs:

- **Codex Cloud**: enable **Common Dependencies** domain allow-list.
- **Claude Code Web**: choose **Custom** domains and check **also include default list of package managers**.

Then add these custom domains (incomplete list):

- `api.github.com`
- `*.blob.core.windows.net`

## Troubleshooting

- **401/403 from GitHub API**: token is missing/invalid, expired, or lacks `Actions: Read`, or you forgot to enable internet access.
- **No artifact found**: workflow has not run successfully on `main` yet.
- **`conda-unpack` not found**: artifact may be stale/corrupt; re-run workflow on `main`.
