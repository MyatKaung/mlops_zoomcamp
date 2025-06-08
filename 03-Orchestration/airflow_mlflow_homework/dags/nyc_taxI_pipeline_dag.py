import os
import logging
from datetime import datetime, timedelta

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
import mlflow
import requests 
import joblib
import numpy as np 

from google.oauth2 import service_account
from google.cloud import storage
import google.auth

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---

# Define a base directory for data storage within the Airflow environment
# This is where raw and processed data files will be stored by the tasks.
# It's local to the Airflow worker container.
_airflow_home_env = os.getenv('AIRFLOW_HOME', '~/airflow')
AIRFLOW_HOME = os.path.expanduser(_airflow_home_env)

# Data directory relative to AIRFLOW_HOME
DATA_DIR_NAME = 'airflow_mlflow_homework_data'
DATA_DIR = os.path.join(AIRFLOW_HOME, 'data', DATA_DIR_NAME)
os.makedirs(DATA_DIR, exist_ok=True) # Ensure the data directory exists

logging.info(f"Data directory for this DAG: {DATA_DIR}")


# Define paths for raw and processed data files within the data directory
RAW_DATA_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet"
RAW_DATA_FILENAME = "yellow_tripdata_2023-03.parquet"
RAW_DATA_PATH = os.path.join(DATA_DIR, RAW_DATA_FILENAME)

PROCESSED_DATA_FILENAME = "yellow_tripdata_2023-03_processed.parquet"
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, PROCESSED_DATA_FILENAME) # Corrected variable name



# --- MLflow Configuration for Server Hosted on GCP VM ---
# Set the tracking URI to the External IP and port of your GCE VM
# This is the network address where the MLflow server application is listening.
# Your Airflow DAG's MLflow client will send requests here.

# Choose an experiment name. MLflow will create this experiment
# in the Cloud SQL database via the MLflow server.
MLFLOW_EXPERIMENT_NAME = "nyc-taxi-self-managed-gcp-server" # Distinct experiment name


MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
MLFLOW_EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME')


# --- Python Functions for Tasks ---

# These functions contain the core data processing logic.
# They are wrapped in Prefect tasks later, or used directly in Airflow.

def download_data_callable(url: str, output_path: str) -> str:
    """
    Downloads data from URL to output_path if it doesn't exist.
    Used by the download task.
    """
    # Use Airflow's logger if running as a task, otherwise script logger
    logger = logging.getLogger(__name__) if 'airflow' not in globals() else logging.getLogger('airflow.task')
    if not os.path.exists(output_path):
        logger.info(f"Downloading data from {url} to {output_path}")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status() # Raise an exception for HTTP errors
            os.makedirs(os.path.dirname(output_path), exist_ok=True) # Ensure parent dir exists
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info("Download complete.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download data: {e}")
            raise
    else:
        logger.info(f"Data already exists at {output_path}. Skipping download.")
    return output_path


def load_and_log_initial_data_callable(raw_data_path: str) -> str:
    """
    Loads raw data and logs the number of records (for Q3).
    Used by the load/log task.
    """
    logger = logging.getLogger(__name__) if 'airflow' not in globals() else logging.getLogger('airflow.task')
    logger.info(f"Reading raw data from {raw_data_path}")
    df = pd.read_parquet(raw_data_path)
    num_records = len(df)
    logger.info(f"Q3. Number of records loaded: {num_records}")
    print(f"Q3. Number of records loaded: {num_records}") # Also print for visibility in Airflow UI logs
    return raw_data_path # Return the path to pass to the next task


def prepare_data_callable(raw_data_path: str, processed_data_path: str) -> str:
    """
    Reads raw data, applies transformations, logs size, and saves processed data.
    Used by the prepare task.
    """
    logger = logging.getLogger(__name__) if 'airflow' not in globals() else logging.getLogger('airflow.task')
    logger.info(f"Preparing data from {raw_data_path}")
    df = pd.read_parquet(raw_data_path)

    # Calculate trip duration in minutes
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    # Filter for durations between 1 and 60 minutes
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    # Convert location IDs to strings
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    num_records_processed = len(df)
    logger.info(f"Q4. Size of the result after preparation: {num_records_processed}")
    print(f"Q4. Size of the result after preparation: {num_records_processed}")

    # Save the processed data
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True) # Ensure parent dir exists
    df.to_parquet(processed_data_path, index=False)
    logger.info(f"Processed data saved to {processed_data_path}")

    return processed_data_path # Return the path to pass to the next task
def initialize_gcp_credentials():
    """Initialize and verify GCP credentials"""
    logger = logging.getLogger('airflow.task')
    
    creds_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    logger.info(f"Using GOOGLE_APPLICATION_CREDENTIALS: {creds_path}")
    
    if not creds_path:
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable not set")
    
    if not os.path.exists(creds_path):
        raise FileNotFoundError(f"Credentials file not found: {creds_path}")
    
    try:
        # Load and verify credentials
        credentials = service_account.Credentials.from_service_account_file(creds_path)
        logger.info(f"Successfully loaded credentials for project: {credentials.project_id}")
        logger.info(f"Service account email: {credentials.service_account_email}")
        
        # Test the credentials by creating a storage client
        storage_client = storage.Client(credentials=credentials, project=credentials.project_id)
        logger.info("Successfully created GCS client with service account credentials")
        
        return credentials
    except Exception as e:
        logger.error(f"Failed to initialize GCP credentials: {e}")
        raise
def train_model_callable(processed_data_path: str) -> str:
    """
    Trains model, logs to the MLflow server on the GCP VM, and prints Q5 & Q6 answers.
    """
    logger = logging.getLogger(__name__) if 'airflow' not in globals() else logging.getLogger('airflow.task')
    logger.info(f"Training model using data from {processed_data_path}")
    
    # Initialize GCP credentials first
    try:
        credentials = initialize_gcp_credentials()
        logger.info("GCP credentials initialized successfully")
        
        # Set credentials in environment for MLflow to use
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        
    except Exception as e:
        logger.error(f"Failed to initialize GCP credentials: {e}")
        raise
    
    df_processed = pd.read_parquet(processed_data_path)

    categorical = ['PULocationID', 'DOLocationID']
    target = 'duration'

    # --- MLflow Logging to Server ---
    logger.info(f"Setting MLflow tracking URI to: {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI) 

    logger.info(f"Setting MLflow experiment to: {MLFLOW_EXPERIMENT_NAME}")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    mlflow.set_registry_uri(MLFLOW_TRACKING_URI) 

    # Start an MLflow run within the experiment
    with mlflow.start_run(run_name="airflow_gcp_server_run") as run:
        run_id = run.info.run_id
        logger.info(f"Started MLflow run with ID: {run_id}")

        # Log parameters
        mlflow.log_param("data_path", processed_data_path)
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("categorical_features", categorical)
        mlflow.log_param("target", target)

        # Prepare data for modeling
        dv = DictVectorizer()
        train_dicts = df_processed[categorical].to_dict(orient='records')
        X_train = dv.fit_transform(train_dicts)
        y_train = df_processed[target].values

        # Train the Linear Regression model
        lr = LinearRegression()
        lr.fit(X_train, y_train)

        # Log the model's intercept as a metric (Q5)
        intercept = lr.intercept_
        logger.info(f"Q5. Intercept of the model: {intercept:.2f}")
        print(f"Q5. Intercept of the model: {intercept:.2f}")
        mlflow.log_metric("intercept", intercept)

        # Calculate and log Train RMSE
        y_pred = lr.predict(X_train)
        rmse = root_mean_squared_error(y_train, y_pred)
        mlflow.log_metric("rmse_train", rmse)
        logger.info(f"Train RMSE: {rmse:.2f}")

        # --- Log Artifacts with explicit GCP client ---
        try:
            # Log the DictVectorizer (preprocessor)
            dv_output_path = "dict_vectorizer.pkl"
            joblib.dump(dv, dv_output_path)
            
            logger.info("Attempting to log DictVectorizer artifact...")
            mlflow.log_artifact(dv_output_path, artifact_path="preprocessor")
            logger.info("Successfully logged DictVectorizer artifact")
            
            # Clean up the temporary local file
            if os.path.exists(dv_output_path):
                os.remove(dv_output_path)
                logger.info(f"Cleaned up local file: {dv_output_path}")

            # Log the trained Linear Regression model
            sample_input_array = X_train[0].toarray() 
            model_name = f"YellowTaxiMarch2023GCPManaged_{run_id[:8]}"


            mlflow.sklearn.log_model(
                sk_model=lr,
                artifact_path="linear_regression_model",
                registered_model_name=model_name, 
                input_example=sample_input_array,  
            )
            logger.info("Successfully logged sklearn model")

        except Exception as e:
            logger.error(f"Failed to log artifacts to MLflow: {e}")
            raise

        # Print instruction for Q6
        print(f"For Q6, check the MLflow UI at {MLFLOW_TRACKING_URI} -> Experiments -> {MLFLOW_EXPERIMENT_NAME} -> Run ID {run_id} -> Artifacts -> linear_regression_model -> MLmodel file size.")

    return run_id


# --- Airflow DAG Definition ---

# Import Airflow components only if running in an Airflow environment
try:
    from airflow.decorators import dag, task
    # from airflow.operators.python import PythonOperator # Not strictly needed with @task decorator
    STATIC_START_DATE = datetime(2023, 1, 1, 0, 0, 0)


    # Define default arguments for the DAG
    DEFAULT_ARGS = {
        'owner': 'airflow_user',
        'depends_on_past': False,
        'start_date': STATIC_START_DATE,
        'email_on_failure': False,
        'email_on_retry': False,
        'retries': 1, # Retry tasks once on failure
        'retry_delay': timedelta(minutes=2), # Wait 2 minutes between retries
    }

    # Define the DAG using the @dag decorator
    @dag(
        dag_id='nyc_taxi_mlflow_gcp_server_dag', # Unique DAG ID for this setup
        default_args=DEFAULT_ARGS,
        schedule="@once",
        catchup=False, # Don't run for past dates between start_date and today
        tags=['homework', 'mlflow', 'nyc-taxi', 'gcp-server'], # Tags for filtering in the UI
        doc_md="""
        ### NYC Taxi ML Homework DAG (MLflow Server on GCP VM)
        ETL and Train ML model for NYC Taxi data (March 2023),
        logging experiment tracking and model artifacts to a self-managed MLflow server
        running on a GCP Compute Engine VM, using Cloud SQL for backend metadata
        and Google Cloud Storage for artifacts.
        """ # Markdown documentation for the DAG
    )
    def nyc_taxi_ml_pipeline_gcp_server():

        # Define tasks using the @task decorator, wrapping the callables
        @task
        def download_data_task():
            """Airflow task to download raw data."""
            return download_data_callable(RAW_DATA_URL, RAW_DATA_PATH)

        @task
        def load_and_log_initial_data_task(raw_data_file_path: str):
            """Airflow task to load and log initial data count (Q3)."""
            return load_and_log_initial_data_callable(raw_data_file_path)

        @task
        def prepare_data_task(raw_data_file_path: str):
            """Airflow task to prepare data (filtering, feature engineering, Q4 size log)."""
            return prepare_data_callable(raw_data_file_path, PROCESSED_DATA_PATH)

        @task
        def train_model_gcp_server_task(processed_data_file_path: str):
            """Airflow task to train the model and log results/artifacts to MLflow server (Q5, Q6)."""
            return train_model_callable(processed_data_file_path)

        # Define the task dependencies and the overall pipeline flow
        raw_data_path_from_download = download_data_task()

        # load_and_log_initial_data runs after download_data.
        # It takes the download task's output (the raw file path) as input.
        raw_data_path_after_load_log = load_and_log_initial_data_task(raw_data_path_from_download)

        # prepare_data runs after load_and_log.
        # It takes the output of load_and_log (which is still the raw file path).
        processed_data_path_result = prepare_data_task(raw_data_path_after_load_log)

        # train_model_gcp_server runs after prepare_data.
        # It takes the output of prepare_data (the processed file path).
        mlflow_run_id = train_model_gcp_server_task(processed_data_path_result)

        # Optional: Add more tasks here, potentially dependent on mlflow_run_id

    # Instantiate the DAG
    # This makes the DAG available for Airflow to discover
    homework_dag_gcp_server = nyc_taxi_ml_pipeline_gcp_server()

# Handle ImportError if Airflow modules are not available (e.g., running script directly)
except ImportError as e:
    logging.warning(f"Airflow specific modules not found (ImportError: {e}). DAG will not be defined if not in Airflow context.")
    logging.warning("This is expected if running the script outside of an Airflow environment.")

    # --- Local Testing Block (Optional) ---
    # This block allows you to run the callables directly for testing purposes
    # outside of the Airflow scheduler. It does NOT use Airflow orchestration.
    # It will attempt to log to the MLflow server specified by MLFLOW_TRACKING_URI
    # if it's running and accessible from where you run this script.
    if __name__ == "__main__":
        logging.info("\n--- Running script outside of Airflow context for testing functions ---")

        # Create a temporary data directory for local testing
        # Note: This data path is different from the DATA_DIR used *inside* Airflow tasks
        LOCAL_TEST_DATA_DIR = os.path.join(os.path.expanduser("~"), "temp_airflow_homework_data")
        os.makedirs(LOCAL_TEST_DATA_DIR, exist_ok=True)
        LOCAL_RAW_DATA_PATH = os.path.join(LOCAL_TEST_DATA_DIR, RAW_DATA_FILENAME)
        LOCAL_PROCESSED_DATA_PATH = os.path.join(LOCAL_TEST_DATA_DIR, PROCESSED_DATA_FILENAME)
        logging.info(f"Local test data will be stored in: {LOCAL_TEST_DATA_DIR}")

        logging.info("\n--- Testing Download ---")
        downloaded_file = download_data_callable(RAW_DATA_URL, LOCAL_RAW_DATA_PATH)

        logging.info("\n--- Testing Q3 ---")
        load_and_log_initial_data_callable(downloaded_file)

        logging.info("\n--- Testing Q4 ---")
        processed_file = prepare_data_callable(downloaded_file, LOCAL_PROCESSED_DATA_PATH)

        logging.info("\n--- Testing Q5 & Q6 (MLflow Logging) ---")
        logging.info(f"Attempting to log to MLflow server at: {MLFLOW_TRACKING_URI}")
        print(f"Ensure MLflow UI is running and accessible at {MLFLOW_TRACKING_URI} (or adjust for local test)")

        # This will call the train_model_callable directly
        # It will attempt to log to the server defined by MLFLOW_TRACKING_URI
        try:
            train_model_callable(processed_file)
            logging.info("\n--- Local MLflow Logging Test Complete ---")
            print(f"\nMLflow Experiment: {MLFLOW_EXPERIMENT_NAME}")
            print(f"Check MLflow UI at {MLFLOW_TRACKING_URI} for results and Q6 answer.")
        except Exception as e:
            logging.error(f"\n--- Local MLflow Logging Test Failed: {e} ---", exc_info=True)
            print(f"\nLocal MLflow logging test failed. Ensure MLflow server is running and accessible at {MLFLOW_TRACKING_URI} and your local machine has GCP credentials/network access if needed.")
