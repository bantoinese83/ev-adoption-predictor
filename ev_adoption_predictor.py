import logging
import os
import sqlite3

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from rich.progress import Progress

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')


class EVAdoptionPredictor:
    def __init__(self, database, query, test_size=0.2, random_state=42):
        try:
            self.conn = sqlite3.connect(database)
            self.df = pd.read_sql_query(query, self.conn)
            self.conn.close()
        except sqlite3.Error as e:
            logging.error(f"Error connecting to database: {e}")
            print(f"Error connecting to database: {e}")
            return
        except pd.errors.DatabaseError as e:
            logging.error(f"Error querying database: {e}")
            print(f"Error querying database: {e}")
            return

        logging.info(f"Columns in DataFrame: {self.df.columns}")
        print(f"Columns in DataFrame: {self.df.columns}")

        self.test_size = test_size
        self.random_state = random_state
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.best_model = None

    def preprocess_data(self):
        try:
            data_expanded = self.df['data'].str.split(',', expand=True)
            expected_columns = ['VIN', 'County', 'City', 'State', 'Zip', 'Model_Year', 'Make', 'Model',
                                'Electric_Vehicle_Type', 'Clean_Alternative_Fuel_Vehicle_Eligible',
                                'Electric_Range', 'Base_MSRP', 'Legislative_District', 'DOL_Vehicle_ID',
                                'Location', 'Electric_Utility', 'Census_Tract']

            logging.info(f"Sample data after split:\n{data_expanded.head()}")
            print(f"Sample data after split:\n{data_expanded.head()}")

            if data_expanded.shape[1] > len(expected_columns):
                logging.warning(
                    f"Data has {data_expanded.shape[1]} columns, trimming to {len(expected_columns)} columns")
                print(f"Data has {data_expanded.shape[1]} columns, trimming to {len(expected_columns)} columns")
                data_expanded = data_expanded.iloc[:, :len(expected_columns)]
            elif data_expanded.shape[1] < len(expected_columns):
                raise ValueError(f"Expected {len(expected_columns)} columns, but got {data_expanded.shape[1]} columns")

            data_expanded.columns = expected_columns
            data_expanded = data_expanded.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
            data_expanded['Zip'] = pd.to_numeric(data_expanded['Zip'], errors='coerce')
            data_expanded['Model_Year'] = pd.to_numeric(data_expanded['Model_Year'], errors='coerce')
            data_expanded['Electric_Range'] = pd.to_numeric(data_expanded['Electric_Range'], errors='coerce')
            data_expanded['Base_MSRP'] = pd.to_numeric(data_expanded['Base_MSRP'], errors='coerce')
            data_expanded['Census_Tract'] = pd.to_numeric(data_expanded['Census_Tract'], errors='coerce')
            data_expanded.dropna(inplace=True)

            data_expanded['Income'] = data_expanded['Base_MSRP']
            data_expanded['Population_Density'] = 1000
            data_expanded['Government_Incentives'] = 5000
            data_expanded['Income_Population_Interact'] = data_expanded['Income'] * data_expanded['Population_Density']

            X = data_expanded[['Income', 'Population_Density', 'Government_Incentives', 'Income_Population_Interact']]
            y = data_expanded['Electric_Vehicle_Type']
            y = y.apply(lambda x: 1 if 'BEV' in x else 0)

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.test_size,
                                                                                    random_state=self.random_state)
        except KeyError as e:
            logging.error(f"Error in data preprocessing: {e}")
            print(f"Error in data preprocessing: {e}")
            return None, None, None, None
        except ValueError as e:
            logging.error(f"Error in data preprocessing: {e}")
            print(f"Error in data preprocessing: {e}")
            return None, None, None, None

        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_model(self):
        try:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
            ])
            param_grid = {
                'clf__n_estimators': [100, 200, 300],
                'clf__learning_rate': [0.05, 0.1, 0.2],
                'clf__max_depth': [3, 5, 7]
            }
            grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', verbose=1)
            grid_search.fit(self.X_train, self.y_train)
            self.best_model = grid_search.best_estimator_
            logging.info(f"Best Parameters: {grid_search.best_params_}")
            print(f"Best Parameters: {grid_search.best_params_}")
        except Exception as e:
            logging.error(f"Error in model training: {e}")
            print(f"Error in model training: {e}")
            return

    def evaluate_model(self):
        try:
            y_pred = self.best_model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            logging.info(f"Accuracy: {accuracy}")
            print(f"Accuracy: {accuracy}")

            cm = confusion_matrix(self.y_test, y_pred)
            cr = classification_report(self.y_test, y_pred)
            logging.info(f"Confusion Matrix:\n{cm}")
            logging.info(f"Classification Report:\n{cr}")
            print(f"Confusion Matrix:\n{cm}")
            print(f"Classification Report:\n{cr}")

            plt.figure(figsize=(10, 7))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.show()

        except Exception as e:
            logging.error(f"Error in model evaluation: {e}")
            print(f"Error in model evaluation: {e}")
            return

    def save_model(self, filename):
        try:
            joblib.dump(self.best_model, filename)
            logging.info(f"Model saved to {filename}")
            print(f"Model saved to {filename}")
        except Exception as e:
            logging.error(f"Error saving model: {e}")
            print(f"Error saving model: {e}")
            return

    def load_and_test_model(self, filename):
        try:
            loaded_model = joblib.load(filename)
            y_pred_loaded = loaded_model.predict(self.X_test)
            accuracy_loaded = accuracy_score(self.y_test, y_pred_loaded)
            logging.info(f"Accuracy (Loaded Model): {accuracy_loaded}")
            print(f"Accuracy (Loaded Model): {accuracy_loaded}")

            # Additional Testing Code
            conf_matrix_loaded = confusion_matrix(self.y_test, y_pred_loaded)
            class_report_loaded = classification_report(self.y_test, y_pred_loaded)

            print("Confusion Matrix (Loaded Model):")
            print(conf_matrix_loaded)

            print("Classification Report (Loaded Model):")
            print(class_report_loaded)

        except Exception as e:
            logging.error(f"Error loading and testing model: {e}")
            print(f"Error loading and testing model: {e}")
            return


if __name__ == "__main__":
    logging.info("Starting EV Adoption Predictor")

    # Wrap your code in a Progress context
    with Progress() as progress:
        task1 = progress.add_task("[green]Preprocessing data...", total=1)
        predictor = EVAdoptionPredictor('csv_db.sqlite', "SELECT * FROM csvs")
        progress.update(task1, advance=1)

        task2 = progress.add_task("[green]Training model...", total=1)
        X_train, X_test, y_train, y_test = predictor.preprocess_data()
        if X_train is not None:
            predictor.train_model()
            progress.update(task2, advance=1)

        task3 = progress.add_task("[green]Evaluating model...", total=1)
        if X_train is not None:
            predictor.evaluate_model()
            progress.update(task3, advance=1)

        task4 = progress.add_task("[green]Saving model...", total=1)
        if X_train is not None:
            predictor.save_model('ev_adoption_model.pkl')
            progress.update(task4, advance=1)

        task5 = progress.add_task("[green]Loading and testing model...", total=1)
        if X_train is not None:
            predictor.load_and_test_model('ev_adoption_model.pkl')
            progress.update(task5, advance=1)

    logging.info("Done")
