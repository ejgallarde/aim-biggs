# mlops_final_project

## Project Architecture
![3](images/archi.png)
The **Docker-based architecture** enables end-to-end machine learning workflows by orchestrating components within a single network. Data is fetched using Dagster, and experiments along with trained models are stored in MLflow. MLflow serves the trained models via FastAPI, providing an API for real-time predictions. The API can be accessed through JupyterLab or cURL for local testing, ensuring seamless model deployment and interaction.  

The **GitHub Actions CI/CD pipeline** ensures code quality and reliability through automated checks and testing. Pre-commit hooks enforce formatting and linting standards using tools like Flake8, Black, and Isort. Pytest validates functionality, and successful tests allow changes to be merged, maintaining a robust and well-tested codebase.

## Directory Structure
- `.github/workflows` - contains `pre-commit.yml` defines what is run in github actions
- `images` - image reference to markdown files
- `src` - source directory
  - `api` - fastapi source code to predict from deployed model
  - `dagster` - train model and register model to mlflow
  - `notebooks` - notebook to test 
  - `dockerfiles` - all container dockerfiles consolidated here
- `tests` - test directory

## Github Actions
- pre-commit
  - See `.github/workflows/pre-commit.yml` and `.pre-commit-config.yaml`
- pytest
  - See `tests/*` directory

## Deploying Docker Locally
- Deploying:
```commandline
docker-compose --env-file env_DEV up --build -d
```
- Cleanup:
```commandline
docker-compose down
```

## Executing dagster job
- Select job
![2](images/job_run1.png)
- Run job
![3](images/job_run2.png)
- Waterfall view job summary
  - has 2 flows for normal and skewed data for evidently comparison
![4](images/job_summary.png)

## Model Artifacts
Includes shap plots and evidently reports
![5](images/report1.png)
![6](images/report2.png)
![7](images/report3.png)

## Important links 
- (after containers are up and dagster job has run to test endpoint)
  - dagster [http://localhost:3000](http://localhost:3000)
  - mlflow [http://localhost:5000](http://localhost:5000)
  - minio object storage [http://localhost:9001](http://localhost:9001)
  - jupyterlab [http://localhost:8888](http://localhost:8888)
  - fastapi
    - [http://localhost:8000/docs](http://localhost:8000/docs)
    - to get predictions via cURL (or use notebook in jupyterlab)
    ```commandline
    curl --location 'http://localhost:8000/predict' \
    --header 'Content-Type: application/json' \
    --data '{
        "data": [
            [5.1, 3.5, 1.4, 0.2],
            [6.7, 3.1, 4.4, 1.4],
            [6.7, 3.1, 4.4, 1.4]
        ]
    }'
    ```
    