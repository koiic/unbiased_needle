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

