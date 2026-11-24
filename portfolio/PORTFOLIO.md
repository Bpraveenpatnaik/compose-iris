---
owner: Bpraveenpatnaik
github: https://github.com/Bpraveenpatnaik
program: PGCSEDS, IIIT Hyderabad (2021)
focus: [MLOps, DevSecOps, Docker microservices, Machine learning, PySpark]
projects: 4
---

# Project Portfolio: Praveen Patnaik

Four projects from 2021, built during the **PGCSEDS program at IIIT Hyderabad**. They go from training ML models to serving them as APIs, splitting them into containers, and running automated tests, coverage checks and security scans before release.

| # | Project | Theme | Period | Key tech |
|---|---|---|---|---|
| 1 | [DevSecOps Pipeline for an ML API](#1-devsecops-pipeline-for-an-ml-inference-api) | DevSecOps, CI/CD | Sep 2021 | GitHub Actions, Docker, Anchore, SARIF, Codecov |
| 2 | [Containerized ML Microservices](#2-containerized-ml-microservices-with-docker-compose) | Microservices, Docker | Aug 2021, updated 2026 | Docker Compose, FastAPI, scikit-learn, pytest, GitHub Actions |
| 3 | [MLOps: Iris Prediction API](#3-mlops-iris-prediction-api-with-ci) | MLOps, CI | Jul 2021 | FastAPI, scikit-learn, pytest, GitHub Actions |
| 4 | [House Price Prediction Hackathon](#4-house-price-prediction-hackathon) | Data science, Big data | Sep 2021 | pandas, scikit-learn, PCA, PySpark MLlib |

> **Note:** Projects 1–3 are course assignments. Each one starts from a template repo provided by the course, and Praveen forked it and extended it. The sections below describe his own changes. Project 4 is his own work from start to finish.

### Skills across the projects
- **ML engineering:** model training and evaluation, serving models over REST, online retraining (a feedback loop)
- **MLOps / DevOps:** GitHub Actions (matrix builds, job dependencies, artifacts), Docker, Docker Compose, shared volumes for models
- **DevSecOps:** container vulnerability scanning (Anchore/Grype), SARIF reports, Dependabot
- **Quality:** pytest, FastAPI TestClient, pytest-cov, Codecov
- **Data science:** exploratory data analysis (EDA), correlation analysis, handling missing values, one-hot encoding, PCA, regression, cross-validated RMSE
- **Big data:** PySpark DataFrames, VectorAssembler, Spark MLlib regression
- **Languages and libraries:** Python, FastAPI, Pydantic, Uvicorn, pandas, NumPy, scikit-learn, matplotlib, seaborn

---

## 1. DevSecOps Pipeline for an ML Inference API
**Repo:** https://github.com/Bpraveenpatnaik/devsecops-iris · **Period:** Sep 2021 · **Type:** Course assignment (forked template)

### Summary
Built out a GitHub Actions CI/CD pipeline that tests a containerized FastAPI machine-learning service, measures its code coverage, scans it for security issues and packages it for release. The tests run on 3 Python versions and 2 operating systems.

### Context
A hands-on "shift-left security" assignment: run tests, coverage tracking and container vulnerability scanning automatically on every push, before anything is released. The starter repo had a FastAPI service with a scikit-learn Gaussian Naive Bayes model for the Iris dataset (`/ping`, `/predict_flower`, `/feedback_loop`), a Dockerfile and a basic pipeline.

### What he did
| Area | Contribution |
|---|---|
| Cross-platform CI matrix | Extended the coverage job to a **Python × OS matrix** (3.7, 3.8 and 3.9 × Ubuntu and Windows), so every push runs **6 parallel jobs**. |
| Code coverage | Connected his fork to **Codecov** (`pytest-cov` plus `codecov-action`) and added the repo's coverage badge to the README. |
| Container security | Ran the **Anchore/Grype** scan of the Docker image built in CI, with a medium-severity cutoff, and saved the **SARIF** report as a build artifact. |
| Reporting | Committed the SARIF report and reviewed it in Microsoft's SARIF Web Validator (screenshot in `reports/`). |

### Pipeline
```
push ──► unittest ─────────────┐
     ├─► codecoverage (py3.7/3.8/3.9 × ubuntu/windows) ─► Codecov
     ├─► container-security (Buildx ► Anchore/Grype scan ► SARIF artifact)
     └───────────────────────────► upload_zip (only after all 3 pass) ► release.zip
```

### Security findings
The scan found **72 vulnerabilities** in the `python:3.8-slim` image:

| Severity | Count | Notable |
|---|---|---|
| Critical | 2 | CVE-2021-33574 (glibc) |
| Medium | 6 | CVE-2021-37750 (Kerberos), CVE-2021-40528 (libgcrypt), CVE-2011-3389 (GnuTLS) |
| Low | 6 | n/a |
| Negligible | 58 | n/a |

**Insight:** every finding came from the Debian base image, not the application code, and none had an upstream fix when the scan ran. Ways to reduce the risk: rebuild on a newer base image regularly, use distroless or slimmer images, and remove unneeded layers.

### Resume bullets
- Extended a GitHub Actions CI/CD pipeline for a containerized FastAPI ML service to a cross-platform matrix (3 Python versions × Ubuntu/Windows, 6 parallel jobs), with Codecov coverage reporting.
- Added container vulnerability scanning with Anchore/Grype and SARIF reports, and triaged 72 CVEs in the base image (2 critical, 6 medium).
- Set up gated release packaging, so artifacts are only published after the tests, coverage job and security scan all pass.

---

## 2. Containerized ML Microservices with Docker Compose
**Repo:** https://github.com/Bpraveenpatnaik/compose-iris · **Period:** Aug 2021 · **Type:** Course assignment (forked template)

### Summary
An ML system split into three containerized FastAPI microservices: **preprocessing → training → prediction**. They run together under Docker Compose, talk to each other over HTTP and share trained models through a named volume. The template used the Iris dataset; this version predicts the **cultivar of a wine from all 13 chemical measurements in the UCI Wine dataset**, with a scaled **logistic regression** model (about 98% cross-validated accuracy).

### Architecture
```
client ─POST /process─► processr :8888 ─POST /train─► trainr :7777 ─POST /reload_model─► predictr :9999 ◄─POST /predict_wine─ client
                                                         │                                  ▲
                                                         └──── ./models (shared volume) ─────┘
```
- **processr:** validates and cleans incoming labelled records, then forwards them to `trainr`.
- **trainr:** loads the current model from the shared volume, retrains it, saves it, and tells `predictr` to reload.
- **predictr:** serves predictions from the latest saved model.
- Services find each other through environment variables (`TRAINR_ENDPOINT`, `PREDICTR_ENDPOINT`) and can be scaled with `deploy.replicas`.

### What was built on top of the template
- **New dataset:** moved all three services from Iris to the **UCI Wine dataset**, using all 13 features and the correct cultivar labels, with a `/predict_wine` endpoint.
- **Shared schemas:** a single Pydantic module (`common/wine.py`) used by every service. It rejects negative values and unknown class labels.
- **Better retraining:** feedback is stored and the model is retrained on the base data plus all feedback. The template refit on only the newest batch, which threw away earlier learning.
- **Better model:** `StandardScaler` + logistic regression, reporting 5-fold cross-validated accuracy after each retrain.
- **Reliability:** trainr trains a baseline model on startup. Services return a clear `502` or `503` when a downstream service or the model is unavailable, and requests have timeouts.
- **Containers:** Python 3.11 slim images that run as a non-root user, a named volume for models, and no model files committed to git.
- **Testing and CI:** 11 pytest tests (98% coverage), and a GitHub Actions pipeline that runs them, then builds the Compose stack and checks it end to end with real HTTP requests.

### Resume bullets
- Built a 3-service ML system (preprocess → train → predict) on Docker Compose. The services communicate over REST and share models through a named volume, so retrained models go live without downtime.
- Designed an online-retraining loop that keeps all past feedback and reports cross-validated accuracy (about 98%) after each update.
- Added a pytest suite (98% coverage) and a GitHub Actions pipeline that builds the Compose stack and tests it end to end.

---

## 3. MLOps: Iris Prediction API with CI
**Repo:** https://github.com/Bpraveenpatnaik/mlops-iris · **Period:** Jul 2021 · **Type:** Course assignment (forked template)

### Summary
A FastAPI service that serves an Iris flower classifier, with automated tests and a GitHub Actions pipeline that runs pytest and packages the code as a release artifact. It also has a feedback-loop endpoint for retraining the model on new data.

### What he did
- **Compared models:** added a **Decision Tree** classifier, trained and evaluated alongside the existing Gaussian Naive Bayes model.
- **Tests:** wrote pytest cases for the *Setosa* and *Versicolour* classes using FastAPI's `TestClient`.
- **Response metadata:** added a training-date log and a `timestamp` field to the prediction response schema.
- **Pipeline:** on every push and PR, the CI job runs the tests, then a gated job zips the code and uploads it as an artifact.

### Resume bullets
- Extended a FastAPI ML inference service with a Decision Tree classifier for comparison against Gaussian Naive Bayes, and added class-level API tests that run in GitHub Actions.

---

## 4. House Price Prediction Hackathon
**Repo:** https://github.com/Bpraveenpatnaik/IIITH-Hackathon-2 · **Period:** Sep 2021 · **Type:** Hackathon (IIIT Hyderabad), individual work

### Summary
A full regression workflow for predicting house sale prices on the Ames housing dataset (the data used in Kaggle's "House Prices" competition). It covers EDA, feature engineering, scikit-learn models and a **PySpark MLlib** version of the pipeline.

### Approach
1. **EDA:** looked at the distribution of the target (`SalePrice`), made a correlation heatmap and picked out the top 10 features most correlated with price (e.g. `OverallQual`, `GrLivArea`, `GarageCars`, `TotalBsmtSF`), plus pair plots.
2. **Data cleaning:** built a table of missing values and dropped 18 columns with many gaps (e.g. `PoolQC`, `Alley`, `Fence`, the garage and basement detail columns), then one-hot encoded the categorical columns.
3. **Modelling (scikit-learn):** used linear regression with an 80/20 split and **10-fold cross-validated RMSE**, and saved the model with pickle.
4. **Dimensionality reduction:** used **PCA** to cut the features to 5 components, then compared regression on those components with the full feature set.
5. **Big-data pipeline (PySpark):** set up Spark 3.1 in Colab, loaded the CSV into Spark DataFrames, converted the column types, assembled the features with `VectorAssembler`, and trained an **elastic-net linear regression** (MLlib) on an 80/20 random split.

### Resume bullets
- Built a house-price regression pipeline (EDA, handling missing data, one-hot encoding, PCA, cross-validated linear regression) in pandas and scikit-learn during an IIIT Hyderabad hackathon.
- Rebuilt the pipeline in PySpark MLlib (VectorAssembler plus elastic-net regression) to show the same workflow could scale to big data.

---

## Links
- GitHub profile: https://github.com/Bpraveenpatnaik
- DevSecOps pipeline: https://github.com/Bpraveenpatnaik/devsecops-iris
- Docker microservices: https://github.com/Bpraveenpatnaik/compose-iris
- MLOps API: https://github.com/Bpraveenpatnaik/mlops-iris
- Hackathon: https://github.com/Bpraveenpatnaik/IIITH-Hackathon-2
