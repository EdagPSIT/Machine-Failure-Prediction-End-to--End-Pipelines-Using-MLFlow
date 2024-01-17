import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = "machine-failure"

list_of_files = [
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/data/__init__.py",
    f"src/{project_name}/data/make_data.py",
    f"src/{project_name}/features/__init__.py",
    f"src/{project_name}/features/processing.py",
    f"src/{project_name}/features/transformation.py",
    f"src/{project_name}/features/common_transformation.py",
    f"src/{project_name}/models/__init__.py",
    f"src/{project_name}/models/train.py",
    f"src/{project_name}/models/predict.py",
    f"src/{project_name}/models/evaluate.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/helper_function.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/core.py",
    "data/raw/test.csv",
    "data/processed/test.csv",
    "data/external/test.csv",
    "models/test.pkl",
    "notebook/test.ipynb",
    "configfiles/config.yaml",
    "configfiles/params.yaml",
    "configfiles/schema.yaml",
    "main.py",
    "app.py",
    "requirements.txt",
    "setup.py",
    'README.md',
    "templates/index.html",
    "static/style.css"
]


for filepath in list_of_files:
    filepath = Path(filepath)

    filedir, filename = os.path.split(filepath)

    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")


    else:
        logging.info(f"{filename} is already exists")
