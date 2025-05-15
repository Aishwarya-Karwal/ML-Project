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

