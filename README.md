## End to end ML Project 

The project uses the structure we can commonly call as - "ML Project Pipeline with Modular Components"
(or simply Python Package-based ML Workflow)

This structure follows clean code principles, splitting responsibilities into:

- components/: core ML steps (ingestion, transformation, training)
- pipeline/: orchestration
- utils.py, logger.py, exception.py: support modules
- setup.py: installable as a package (pip install -e .)

- Create a virtual env say - venv , using  conda create -p venv python==3.12
- Activate using - conda activate venv/ (but make sure you are in the project folder)
- To run a script use the command - python filename


### Summary of Project
This project is an End-to-End Machine Learning Pipeline for predicting student exam performance. The pipeline covers all stages from data ingestion to model deployment via a web interface.

#### Project Steps and Components
##### Project Structure Setup

Organized the project into modular folders: components, pipeline, src/utils, config, artifacts, and logs.
Created configuration files and utility scripts for reusability.

#### Data Ingestion

Implemented a data ingestion component to read raw data, save it, and split it into training and testing sets.
Logged each step for traceability.

#### Data Transformation

Built a transformation pipeline using scikit-learn’s ColumnTransformer for preprocessing numerical and categorical features.
Saved the preprocessor object for later use in inference.

#### Model Training

Defined multiple regression models (Linear Regression, Decision Tree, Random Forest, XGBoost, CatBoost, AdaBoost, etc.).
Loaded hyperparameter grids from a YAML config file.
Implemented model evaluation and hyperparameter tuning using GridSearchCV.
Selected and saved the best-performing model.

#### Utility Functions

Created utility functions for saving/loading objects, model evaluation, and hyperparameter tuning.
Ensured robust exception handling and logging throughout.

#### Prediction Pipeline

Developed a prediction pipeline to load the saved preprocessor and model, transform new input data, and generate predictions.
Created a CustomData class to structure user input for prediction.

#### Web Application (Flask)

Built a simple Flask web app for user interaction.
Designed a frontend form for users to input student data and receive predictions.
Handled form submissions, displayed predictions, and ensured user-friendly error messages.

#### Logging and Debugging

Implemented detailed logging for each step to aid in debugging and monitoring.
Addressed and fixed issues related to file paths, column name mismatches, and data formatting.

#### Testing and Validation

Verified the pipeline by running end-to-end tests, checking logs, and validating predictions via the web interface.