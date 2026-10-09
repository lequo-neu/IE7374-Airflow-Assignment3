"""
Customer Segmentation Pipeline (IE7374 Airflow Assignment).

Based on Airflow_Labs/Lab_1. The original DAG was a straight line of four
tasks. This DAG adds a validation gate, two model selection methods that
run in parallel, a branching quality gate, and a run report that executes
no matter which branch was taken.

Flow:
  load -> validate -> preprocess -> [elbow, silhouette] -> select_k -> quality_gate
  quality_gate -> train_final_model -> predict -> report   (silhouette >= threshold)
  quality_gate -> quality_alert ---------------> report   (silhouette <  threshold)
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.standard.operators.python import BranchPythonOperator, PythonOperator

from src.pipeline import (
    compute_elbow,
    compute_silhouette,
    load_data,
    predict_clusters,
    preprocess_data,
    quality_gate,
    select_k,
    train_final_model,
    validate_data,
    write_report,
)

SILHOUETTE_THRESHOLD = 0.25

default_args = {
    "owner": "Kevin Le",
    "start_date": datetime(2026, 10, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}


def _quality_alert(selection: dict) -> dict:
    """Runs only when the quality gate fails. Logs a clear warning for the operator."""
    message = (
        f"Clustering rejected: silhouette {selection['silhouette']} is below "
        f"{SILHOUETTE_THRESHOLD}. The previous model (if any) was kept."
    )
    print(f"WARNING: {message}")
    return {"alert": message}


def _build_report(ti, run_id, **_) -> str:
    """Collect results from every upstream task (skipped tasks return None) and save them."""
    report = {
        "run_id": run_id,
        "validation": ti.xcom_pull(task_ids="validate_data_task"),
        "selection": ti.xcom_pull(task_ids="select_k_task"),
        "final_model": ti.xcom_pull(task_ids="train_final_model_task"),
        "predictions": ti.xcom_pull(task_ids="predict_task"),
        "alert": ti.xcom_pull(task_ids="quality_alert_task"),
    }
    report["status"] = "model_updated" if report["final_model"] else "model_rejected"
    return write_report(report, run_id)


with DAG(
    dag_id="Customer_Segmentation_Pipeline",
    default_args=default_args,
    description="KMeans customer segmentation with validation, parallel model selection and a quality gate",
    schedule=None,
    catchup=False,
    tags=["ie7374", "assignment", "clustering"],
) as dag:

    load_data_task = PythonOperator(
        task_id="load_data_task",
        python_callable=load_data,
    )

    validate_data_task = PythonOperator(
        task_id="validate_data_task",
        python_callable=validate_data,
        op_args=[load_data_task.output],
    )

    preprocess_task = PythonOperator(
        task_id="preprocess_task",
        python_callable=preprocess_data,
        op_args=[load_data_task.output],
    )

    elbow_task = PythonOperator(
        task_id="elbow_task",
        python_callable=compute_elbow,
        op_args=[preprocess_task.output],
    )

    silhouette_task = PythonOperator(
        task_id="silhouette_task",
        python_callable=compute_silhouette,
        op_args=[preprocess_task.output],
    )

    select_k_task = PythonOperator(
        task_id="select_k_task",
        python_callable=select_k,
        op_args=[elbow_task.output, silhouette_task.output],
    )

    quality_gate_task = BranchPythonOperator(
        task_id="quality_gate_task",
        python_callable=quality_gate,
        op_args=[select_k_task.output],
        op_kwargs={"threshold": SILHOUETTE_THRESHOLD},
    )

    train_final_model_task = PythonOperator(
        task_id="train_final_model_task",
        python_callable=train_final_model,
        op_args=[preprocess_task.output, select_k_task.output],
    )

    quality_alert_task = PythonOperator(
        task_id="quality_alert_task",
        python_callable=_quality_alert,
        op_args=[select_k_task.output],
    )

    predict_task = PythonOperator(
        task_id="predict_task",
        python_callable=predict_clusters,
    )

    report_task = PythonOperator(
        task_id="report_task",
        python_callable=_build_report,
        trigger_rule="none_failed_min_one_success",
    )

    load_data_task >> validate_data_task >> preprocess_task
    preprocess_task >> [elbow_task, silhouette_task] >> select_k_task
    select_k_task >> quality_gate_task >> [train_final_model_task, quality_alert_task]
    train_final_model_task >> predict_task >> report_task
    quality_alert_task >> report_task


if __name__ == "__main__":
    dag.test()
