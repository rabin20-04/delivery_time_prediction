import mlflow
import dagshub
import json
from pathlib import Path
from mlflow import MlflowClient
import logging


logger = logging.getLogger("register_model")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

logger.addHandler(handler)

formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

import dagshub # type: ignore

dagshub.init(repo_owner="rabin20-04", repo_name="delivery_time_prediction", mlflow=True)


mlflow.set_tracking_uri(
    "https://dagshub.com/rabin20-04/delivery_time_prediction.mlflow"
)


def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
        
    return run_info


if __name__ == "__main__":

    root_path = Path(__file__).parent.parent.parent
    
    run_info_path = root_path / "run_information.json"
    
    run_info = load_model_information(run_info_path)
    
    run_id = run_info["run_id"]
    model_name = run_info["model_name"]
    
    model_registry_path = f"runs:/{run_id}/{model_name}"
    
    
    model_version = mlflow.register_model(model_uri=model_registry_path,
                                          name=model_name)
    
    
    registered_model_version = model_version.version
    registered_model_name = model_version.name
    logger.info(f" latest model version in model registry is {registered_model_version}")
    
    client = MlflowClient()
    client.transition_model_version_stage(
        name=registered_model_name,
        version=registered_model_version,
        stage="Staging"
    )
    
    logger.info("Model pushed to Staging stage")
    