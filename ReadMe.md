
# Disaster Response Pipeline Project



In this project, it is expected to build a model for an API that classifies disaster messages. The disaster data is provided by Figure Eight.

Required dataset containing real messages that were sent during disaster events is included in the **data** folder.

### Instructions:
The first part of your data pipeline is the Extract, Transform, and Load process. Here, you will read the dataset, clean the data, and then store it in a SQLite database. 

1. Run the following commands in the project's root directory to set up your database and model.
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
        
The second part is for the machine learning portion; splitting the data into a training set and a test set, then, creating a machine learning pipeline that uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV to output a final model that uses the message column to predict classifications for 36 categories (multi-output classification). The last part is exporting the model to a pickle file.


2. Run the following commands in the project's root directory to set up your database and model.
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`


3. Run the following command in the app's directory to run your web app.
    `python run.py`
    
    Your web app should now be running if there were no errors.
    
    Now, open another Terminal Window and Type
    
    env|grep WORK
    
    In a new web browser window, type in the following: https://SPACEID-3001.SPACEDOMAIN


```python

```
