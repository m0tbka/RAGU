import json
import os
import openai
import logging
import pandas as pd

from ragu.common.global_parameters import logs_dir, current_time, run_output_dir


openai._utils._logs.logger.setLevel(logging.WARNING)
openai._utils._logs.httpx_logger.setLevel(logging.WARNING)


log_filename = os.path.join(logs_dir, f"ragu_logs_{current_time}.log")
logging.basicConfig(
    level=logging.INFO,
    filename=log_filename,
    filemode="w",
    format="%(asctime)s %(levelname)s %(message)s"
)

def log_outputs(df: pd.DataFrame, filename: str):
    """
    Save DataFrame in specified directory.

    :param df: DataFrame to save.
    :param filename: filename for the saved file.
    """
    filepath = os.path.join(run_output_dir, f"{filename}.parquet")
#    df.to_parquet(filepath, index=False, engine='pyarrow')
    logging.info(f"Outputs saved in: {filepath}")

def log_metrics(metrics: dict, filename: str):
    """
    Log metrics in specified directory.

    :param filename:
    :param metrics: metrics to save.
    """
    metrics_filepath = os.path.join(run_output_dir, f"{filename}.json")
    with open(metrics_filepath, "w") as f:
        json.dump(metrics, f)
    logging.info(f"Metrics saved in: {metrics_filepath}")

