# Public deployment guide

**Streamlit app not yet deployed.** Running the app locally does not publish it. GitHub publication and Streamlit hosting are separate steps; a repository URL is not a live app URL.

## Preferred option: Streamlit Community Cloud

Streamlit's official documentation describes Community Cloud as free and connected to GitHub repositories. This is the simplest later path for this existing Streamlit app. Check current service terms and limits before publishing. [Official overview](https://docs.streamlit.io/deploy/streamlit-community-cloud).

The owner authorized publication as part of this deployment-engineering task. Scientific work remains frozen. **No explicit reuse license is currently granted.** Authorship is Khadis Aidyn; no email, ORCID or affiliation is invented.

## Preflight

From the repository root, in a separate Python 3.13 environment:

```bash
pip install -r demo/requirements.txt
python scripts/check_demo_deployment.py --history --smoke
streamlit run demo/app.py
```

The check fails nonzero on missing/untracked runtime assets, mismatched evidence, credential-pattern findings or forbidden runtime dependencies. Its credential scan is heuristic, not proof of the absence of secrets. It never prints matched values. The smoke option exercises all 30 cases without scientific modules. No ignored archive is required; full bundle regeneration is a separate optional archival operation, never a startup command.

## GitHub publication

Inspect `git remote -v` and `gh auth status`. The verified account for this release is `hadisaidyn`; the intended new repository name is `InverPINN`. Do not overwrite an existing repository. Publish only after preflight passes and all changes are committed. Use a non-force push. Keep historical commits; an existing ancestor `main` can be fast-forwarded from `scientific-revision` without overwriting its history.

After repository existence and authorization are confirmed:

```bash
git push -u origin main
```

Do not force-add `results/`, `artifacts/`, local environments or `.streamlit/secrets.toml`. The [audit](DEPLOYMENT_AUDIT.md) records prepublication checks; the final task handoff reports the actual push status. An intended repository name is not a claim that creation succeeded.

## Streamlit Community Cloud UI

1. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in using the repository owner's GitHub account.
2. Select **Create app**, then **Yup, I have an app** if prompted.
3. Repository: **`hadisaidyn/InverPINN`**, after the verified GitHub publication has completed.
4. Branch: **`main`**.
5. Main file path: **`demo/app.py`**.
6. Open **Advanced settings**. Choose **Python 3.13**; leave **Secrets** empty. Save.
7. Select **Deploy** and wait for the build. The platform chooses the Python patch version; local testing used 3.13.3. No `runtime.txt`, fixed port or localhost server binding is needed.
8. Open the actual URL assigned by Streamlit. A custom subdomain may be requested only if available; no particular URL is promised here.
9. Verify 30 scenarios, 23/30 J1 versus 27/30 required, 30/30 classical, all seven J1 failures, Q-bias plot, map/metrics changes, source markers, gallery and paper download. Record the real URL only after it loads, then update the README in a separate commit.

Community Cloud defaults to Python 3.12 and supports maintained Python releases; explicitly select 3.13 to match the tested environment. NumPy's pinned version requires ≥3.12; all five pinned dependencies support 3.13, and Linux wheel resolution was checked. [Official runtime/UI instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

Confirm **`demo/requirements.txt`** is selected, not the root research requirements. Community Cloud searches the entry-point directory first. The viewer directory has one requirements file; it excludes the scientific-training stack. [Official dependency rules](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).

No credentials are needed by the app. User login to GitHub/Streamlit is a publication step, not an app secret. Never paste a GitHub token into committed code or into this app's configuration.

The app caches a compact committed bundle, not checkpoints or full reference archives. Startup checks hashes and fails on altered evidence. Cold-start/network behavior and service availability are not guaranteed by the local QA. No hosted deployment was tested in this task.
