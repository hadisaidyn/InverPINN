# Optional public deployment

**Not deployed.** No account, repository publication, credentials or hosting resources were created. Running the app locally does not publish it.

## Preferred option: Streamlit Community Cloud

Streamlit's official documentation describes Community Cloud as free and connected to GitHub repositories. This is the simplest later path for this existing Streamlit app. Check current service terms and limits before publishing. [Official overview](https://docs.streamlit.io/deploy/streamlit-community-cloud).

Only after the repository owner explicitly authorizes publication:

1. Review authorship, licensing, repository visibility and the synthetic-data disclaimer. This repository currently has no configured remote or license grant; do not invent either.
2. Publish the intended repository to the owner's GitHub account, including `demo/data/`, the paper evidence/generated files and the existing paper PDF. Do not publish ignored research archives, local audit records or secrets.
3. Connect the authorized GitHub repository to Streamlit Community Cloud. Select the intended branch and **`demo/app.py`** as the entry point. Use **Python 3.13**, the version tested locally.
4. Confirm that **`demo/requirements.txt`** is selected, not the root research requirements. Community Cloud searches the entry-point directory before the repository root; keeping the viewer dependency file adjacent to the app avoids installing the scientific stack. [Official dependency rules](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).
5. Leave secrets empty: the app needs none. Do not configure a command that rebuilds data, trains, simulates or runs historical experiments.
6. After an authorized deployment, verify all 30 IDs, the 23/30 versus 27/30 failed confirmation, the seven retained failures, plots and local PDF download. Check laptop/tablet widths. Record the actual URL only after it exists.

The app caches a compact committed bundle, not checkpoints or full reference archives. Startup checks hashes and fails on altered evidence. Cold-start/network behavior and service availability are not guaranteed by the local QA. No hosted deployment was tested in this task.
