import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pathlib
import os

from sklearn.neural_network import MLPClassifier    # multi-layered neural network AI model
from sklearn.ensemble import GradientBoostingClassifier # gradient boosting classifier

# dataset_path = 'dataset_sample/data.csv'

def load_data(path):
    # Load the dataset

    # Replace the file name with your actual file name if needed
    df = pd.read_csv(path)
    return df


def create_svc_model():
    # create the SVC model (can be changed with other models down the line)
    model = svm.SVC()
    return model


def create_mlp_classifier():
    # create the mlpclassifier 
    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, activation='relu', solver='adam', random_state=42)
    return model


def create_gradient_boosting_classifier():
    # create the GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    return model


def train_model(model, df):
    # Select the features (sepal_length, sepal_width, petal_length, and petal_width) and species as X
    X = df.drop('target', axis=1)
    y = df['target']
    # Split the dataset into a training set and a test set (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Fit the model to the training data
    model.fit(X_train, y_train)
    
    # Make predictions on the test data
    y_pred = model.predict(X_test)
    
    # print the accuracy score 
    print("Accuracy Report: " + str(accuracy_score(y_test, y_pred)))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix: " + str(confusion_matrix(y_test, y_pred)))
    
    show_confusion_matrix(y_test, y_pred)
    
    return model


def test_model(model, df):
    # Select the features (sepal_length, sepal_width, petal_length, and petal_width) and species as X
    X = df.drop('target', axis=1)
    y = df['target']
    # Split the dataset into a training set and a test set (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Make predictions on the test data
    y_pred = model.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    confusion_matrix(y_test, y_pred)


def show_confusion_matrix(y_test, y_pred):
    # Compute the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Display the confusion matrix using a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["bad", "good"], yticklabels=["bad", "good"])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


# Do prediction with new data
def predict(model, data):
    feature_names = ["frame_time", "ball_center_x", "ball_center_y", "racket_center_x", "racket_center_y", "person_center_x", "person_center_y"]
    data_df = pd.DataFrame(data, columns=feature_names)
    result =[]
    label = ["bad", "good"]
    predicted_value = model.predict(data_df)
    # print('predicted numerical value: ', predicted_value)
    # convert the numerical value back to its original categorical label
    for value in predicted_value:
        result.append(label[value])
    return result


def get_master_csv():
    MASTER_CSV_PATH = "data/master_dataset.csv"
    combined_df = pd.read_csv(MASTER_CSV_PATH)
    
    # rename the category column to target
    # combined_df.rename(columns={'category': 'target'}, inplace=True)
    
    # combined_df['target'] = combined_df['target'].apply(lambda x: 1 if x == "good" else 0)
                
    return combined_df
            
            
def load_csv_files_from_csv():
    # initialize the dataframe
    combined_df = pd.DataFrame()
        
    # iterate through the output csv files
    for filename in os.listdir(OUTPUT_CSV_DIR):
        filepath = os.path.join(OUTPUT_CSV_DIR, filename)
        
        if os.path.isfile(filepath):
            data = pd.read_csv(filepath)
            pd.concat([combined_df, data], ignore_index=True)   # concatenate the datasets to make a really big one!
    
    return combined_df

if __name__ == '__main__':
    x = [[6.0, 0.87, 0.71, 0.94, 0.87, 0.84, 0.8]]
    MASTER_CSV_PATH = "data/master_dataset.csv"
    OUTPUT_CSV_DIR = "data/output_csv_files"
    

    # create the model
    model = create_mlp_classifier()
    
    # load from the master csv
    combined_df = get_master_csv()
    
    # train the model
    model = train_model(model, combined_df)
    
    pred = predict(model, x)
    print(pred)
