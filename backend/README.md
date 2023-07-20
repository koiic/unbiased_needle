### ML MODEL MANAGER

A `.env` file is required to run the application. The file should contain the following variables:

```
GOOGLE_APPLICATION_CREDENTIALS = <path to google credentials json file>
FIREBASE_WEB_API_KEY = <firebase web api key>

AWS_ROLE = <aws role with SageMaker rights>

INSTANCE_TYPE = <aws instance type>
MODEL_DATA_PATH = <aws s3 path to model data

DB_ENGINE_URL = <database engine url>
```

Once the file is created:

```bash
# Move to backend directory
cd backend

# Create virtual environment
pyenv virtualenv 3.10 unbiased_needle

# Activate virtual environment
pyenv activate unbiased_needle

# Install dependencies
pip install -r requirements.txt

# Run application
make run_api
```

To test it:

```bash
# Create a Model
curl -X 'POST' \
  'http://localhost:8000/api/v1/ml/models/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "name": "My Model",
  "description": "My dummy model",
  "datasource_id": 12,
  "tag_names": [
    "CoolerTemp", "BathTemp", "CoolerSwitch", "RefridgentTemp", "CompressorCurrent"
  ]
}'

# Create a Model Version
curl -X 'POST' \
  'http://localhost:8000/api/v1/ml/model_versions/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "name": "My dummy model version",
  "datasource_id": 16,
  "start_datetime": "2022-10-14T00:01:00.000Z",
  "end_datetime": "2023-03-22T15:48:00.000Z",
  "train_test_split": 0.8,
  "model_id": 1
}'

# Update Model Version #1
curl -X 'PATCH' \
  'http://localhost:8000/api/v1/ml/model_versions/1' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "datasource_id": 16,
  "train_test_split": 0.7,
  "algorithm_parameters": {"maio_token":"fzzCJ2rR75djmJtNKP5k66pKfLBzDO", "maio_instance_str": "heineken"}
}'

# Launch training for ModelVersion #1
curl -X 'POST' \
  'http://localhost:8000/api/v1/ml/model_versions/1/train' \
  -H 'accept: application/json' \
  -d ''

# Get status of ModelVersion #1
curl -X 'GET' \
  'http://localhost:8000/api/v1/ml/model_versions/1/status' \
  -H 'accept: application/json'

```

