import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree


def read_and_preprocess_data():
    """Reads the CSV data, encodes categorical variables, and prepares inputs/target.

    Returns:
        tuple: (inputs_n, target)
            inputs_n: DataFrame with numerical features
            target: Series with target variable
    """

    df = pd.read_csv("salaries.csv")

    # Separate features (inputs) and target variable
    inputs = df.drop('salary_more_than_100k', axis='columns')
    target = df['salary_more_than_100k']

    # Encode categorical variables
    for col in inputs.select_dtypes(include=['object']):
        le = LabelEncoder()
        inputs[col + '_n'] = le.fit_transform(inputs[col])
        inputs.drop(col, axis='columns', inplace=True)  # Drop original categorical columns

    return inputs, target


def train_and_evaluate_model(inputs_n, target):
    """Trains a decision tree model and evaluates its performance."""

    model = tree.DecisionTreeClassifier()
    model.fit(inputs_n, target)

    accuracy = model.score(inputs_n, target)
    print("Model accuracy:", accuracy)

    # Make predictions on new data
    new_data = pd.DataFrame([[2, 1, 0], [2, 1, 1]], columns=['company_n', 'job_n', 'degree_n'])  # Add column names
    predictions = model.predict(new_data)
    print("Predictions:", predictions)


if __name__ == '__main__':
    inputs_n, target = read_and_preprocess_data()
    train_and_evaluate_model(inputs_n, target)
