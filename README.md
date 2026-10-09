# IE7374 Lab Assignment: Airflow Customer Segmentation Pipeline

## Overview

This project orchestrates a customer segmentation workflow with Apache Airflow
running in Docker. It is based on Airflow_Labs/Lab_1 from the course repository,
which trains a KMeans model on credit card customer data and picks the number of
clusters with the elbow method.

The original lab is a straight line of four tasks. This version turns it into a
DAG of eleven tasks that validates the data before spending any compute, runs two
model selection methods in parallel, decides through a branching quality gate
whether the new model is good enough to keep, and writes a JSON report at the end
of every run regardless of which branch was taken.

```
load -> validate -> preprocess -> [elbow, silhouette] -> select_k -> quality_gate
quality_gate -> train_final_model -> predict -> report      (silhouette >= 0.25)
quality_gate -> quality_alert ------------------> report      (silhouette <  0.25)
```

![DAG run grid view](assets/dag_grid_view.png)

## Changes Made to the Lab

**DAG structure (dags/customer_segmentation_dag.py)**

This is the main change, because the DAG itself is the core concept of the lab.

A validation task was added right after loading. It stops the run early if any
required column is missing, if there are fewer than 1,000 rows, or if any feature
has more than 5% nulls. In the original lab, bad data would only surface as a
confusing error deep inside model training.

Model selection now fans out into two tasks that run at the same time. One
computes the elbow of the SSE curve, the other computes the silhouette score for
every k from 2 to 15. A third task fans the results back in and picks the final k.

A BranchPythonOperator acts as a quality gate. If the silhouette score at the
chosen k is at least 0.25, the DAG continues to train and save the final model and
then predicts clusters for new customers. If not, it takes the alert branch, logs
a warning, and leaves the previously saved model untouched.

A report task closes every run. It uses the trigger rule
none_failed_min_one_success, so it runs whether the DAG took the training branch
or the alert branch, and it collects the results of every upstream task into one
JSON file under dags/reports.

Default arguments were changed from zero retries to one retry with a one minute
delay, so a transient failure does not fail the whole run.

**Pipeline logic (dags/src/pipeline.py)**

All task logic lives in a separate module with no Airflow imports, so each
function can be tested with plain pytest without starting Airflow. Data still
travels between tasks through XCom as base64 encoded pickles, the same technique
the original lab uses.

The feature set grew from three columns to five: BALANCE, PURCHASES,
CASH_ADVANCE, CREDIT_LIMIT and PAYMENTS. These amounts are heavily right skewed,
so they are now log transformed and standardized instead of min max scaled, which
keeps a few very large accounts from dominating the distance calculation. Nulls
are dropped only in the selected features rather than across all eighteen
columns, which keeps more rows.

**Two bugs fixed from the original lab**

The original code trained KMeans for k from 1 to 49 inside a loop and saved
whatever model was fitted last, so the saved model always had 49 clusters
regardless of what the elbow method recommended. This version retrains a fresh
model with the selected k and saves that one.

The original code also trained on scaled data but predicted on raw, unscaled test
data. This version saves the fitted scaler together with the model in one bundle
and applies the exact same transformation before predicting.

**Tests (test/test_pipeline.py)**

Thirteen unit tests cover validation failures, preprocessing, both selection
methods, the k selection rule, both quality gate outcomes, a full train and
predict round trip, and report writing.

## Advanced Extension: Parallel Model Selection with a Branching Quality Gate

**Why this was chosen**

The original DAG trains a model, prints a recommendation, and stops. Nothing in
the workflow decides anything, so a poor clustering would be saved and used just
as readily as a good one. That is the gap between a script that happens to run on
a schedule and a pipeline that can be trusted to run unattended. In a real
segmentation system, a model that splits customers into meaningless groups is
worse than keeping last week's model, because downstream teams act on the
segments.

The elbow method on its own is also a weak basis for that decision. It looks only
at how much error drops as k grows and says nothing about whether the resulting
clusters are actually well separated. Silhouette measures exactly that, but it
can favour very small k. Using both gives a sturdier answer than either alone,
and since they do not depend on each other, Airflow can run them side by side.

**How it works**

After preprocessing, elbow_task and silhouette_task start in parallel from the
same input. Each fits KMeans for every k from 2 to 15. select_k_task waits for
both and applies a simple rule: keep the elbow k if its silhouette score is
within 0.05 of the best silhouette score, otherwise switch to the silhouette
winner. The reason for the decision is recorded alongside the result.

quality_gate_task then reads the silhouette at the chosen k and returns the id
of the next task to run. Airflow marks the branch that was not chosen as
skipped, and the report task still runs at the end because its trigger rule only
requires that nothing failed and at least one upstream task succeeded.

**What it produces**

Every run leaves a self explanatory record in dags/reports. It shows the
validation summary, which k each method suggested, which k was chosen and why,
the silhouette score that the gate judged, the size of every cluster, the
predicted clusters for new customers, and a status of either model_updated or
model_rejected. In the Airflow grid view, a passing run shows every task green
except quality_alert_task, which is marked skipped, so the path the run took is
visible at a glance.

## Prerequisites

Docker Desktop, with at least 4 GB of memory allocated to Docker (8 GB is
recommended). Download from https://www.docker.com/products/docker-desktop and
verify with: docker --version

Git. Download from https://git-scm.com and verify with: git --version

Python 3.10 or later, only needed to run the unit tests outside Docker. Verify
with: python3 --version

The Airflow image is apache/airflow:3.3.2, pinned as the default in
docker-compose.yaml. The setup script tells the containers to also install
scikit-learn and kneed when they start, so nothing needs to be installed inside
Docker by hand.

## Environment Setup

Clone the repository.

```
git clone https://github.com/lequo-neu/IE7374-Airflow-Assignment3.git
cd IE7374-Airflow-Assignment3
```

Run the setup script once. It creates the folders Airflow mounts and writes a
.env file with your user id and the extra Python packages.

```
bash setup.sh
```

Initialize the Airflow database. The first run downloads the images and can take
a few minutes.

```
docker compose up airflow-init
```

## Running the Code

**Optional: run the unit tests first, without Docker**

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pytest -v
```

**Start Airflow and trigger the DAG**

Start all services in the background, wait about a minute, then check that they
report healthy.

```
docker compose up -d
docker compose ps
```

Open http://localhost:8080 in a browser and log in with username airflow and
password airflow. Find Customer_Segmentation_Pipeline in the Dags list, switch it
on with the toggle, and click Trigger.

When the run finishes, check the outputs on your machine.

```
ls dags/model dags/reports
cat dags/reports/*.json
```

Stop everything when you are done.

```
docker compose down
```

If port 8080 is already used by another program, change the host port and open
http://localhost:8081 instead.

```
sed -i '' 's/"8080:8080"/"8081:8080"/' docker-compose.yaml
```

Warnings about FERNET_KEY not being set are expected with the default compose
file and can be ignored for local use.

## What a Successful Run Looks Like

Running the unit tests ends with this line.

```
13 passed in 13.33s
```

In the Airflow grid view, all eleven tasks are green except quality_alert_task,
which is marked skipped because the quality gate passed. The whole run takes
about 20 seconds.

The task logs show both methods agreeing on six clusters.

```
Elbow method suggests k = 6
Silhouette method suggests k = 6 (score 0.3205)
Quality gate PASSED: 0.3205 >= 0.25
```

The report written to dags/reports looks like this.

```json
{
  "run_id": "manual__2026-10-09T19:52:22.393811+00:00",
  "validation": {
    "rows": 8950,
    "null_ratio": {
      "BALANCE": 0.0,
      "PAYMENTS": 0.0,
      "PURCHASES": 0.0,
      "CASH_ADVANCE": 0.0,
      "CREDIT_LIMIT": 0.0001
    }
  },
  "selection": {
    "reason": "elbow k kept (silhouette within tolerance)",
    "k_elbow": 6,
    "chosen_k": 6,
    "silhouette": 0.3205,
    "k_silhouette": 6
  },
  "final_model": {
    "k": 6,
    "model_path": "/opt/airflow/dags/model/kmeans_bundle.pkl",
    "cluster_sizes": {
      "0": 1941,
      "1": 251,
      "2": 1824,
      "3": 1600,
      "4": 1587,
      "5": 1746
    }
  },
  "predictions": [4, 0, 2],
  "alert": null,
  "status": "model_updated"
}
```

The three predictions are the cluster assignments for the three new customers in
dags/data/test.csv. The saved model bundle appears at dags/model/kmeans_bundle.pkl.

## Project Structure

```
IE7374-Airflow-Assignment3/
    assets/
        dag_grid_view.png
    dags/
        customer_segmentation_dag.py    (DAG: validation, parallel selection, branching, report)
        data/
            file.csv                    (credit card customers from the original lab)
            test.csv                    (three new customers to segment)
        src/
            __init__.py
            pipeline.py                 (all task logic, no Airflow imports)
    test/
        test_pipeline.py                (13 unit tests)
    docker-compose.yaml                 (official Airflow 3.3.2 compose, example DAGs turned off)
    setup.sh                            (creates folders and .env)
    pytest.ini
    requirements.txt
    .gitignore
    README.md
```

Folders created at runtime, such as logs, config, plugins, dags/model and
dags/reports, are excluded from version control.
