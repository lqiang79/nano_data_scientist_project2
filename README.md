# Disaster Response Pipeline Project

### Instructions:

0. To prepare the workspace of the project

    - Download the project code from the github repository
    - active your local venv in project root path:

        ```bash
        python3 -m venv venv
        ```

    - insstall all requirements

        ```bash
        pip3 install -r requirements.txt
        ```

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
