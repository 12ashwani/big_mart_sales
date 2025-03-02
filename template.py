import os 
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

project_name='big_mart_sales'
list_of_files = [
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/components/model_monitoring.py",
    f"src/{project_name}/pipelines/__init__.py",
    f"src/{project_name}/pipelines/training_pipeline.py",
    f"src/{project_name}/pipelines/prediction_pipeline.py",
    f"src/{project_name}/transformation/__init__.py",
    f"src/{project_name}/transformation/data_cleaning.py",
    f"src/{project_name}/transformation/feature_engineering.py",
    f"src/{project_name}/transformation/encoding.py",
    f"src/{project_name}/exception.py",
    f"src/{project_name}/logger.py",
    f"src/{project_name}/utils.py",
    "main.py",
    "app.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory:{filedir} for the file {filename}")

    if not filepath.exists():
        with open(filepath, 'w') as f:
            f.write("" if filename.endswith(".py") else "")
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")

# Ensure the project is recognized as a package
with open("src/__init__.py", "w") as f:
    f.write("")

with open(f"src/{project_name}/__init__.py", "w") as f:
    f.write("")

logging.info("Project structure setup complete.")
