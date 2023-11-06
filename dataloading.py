""" 
- Author: Annye Braca
- Purpose: Prepare data for machine learning algorithms
- Date: 2023-10-03

"""



# Custom Modules
from common_imports import *
from getdata import DataSlicer

# from classcounter import analyze_class_distribution

# Display a message indicating that the dataset is being loaded
print("Loading Dataset...")


# Define the get_data function
def get_data():
    """
    fetch data ready for training algorithms

    Parameters:
    - X: Feature matrix
    - techniques: DataFrame containing target variables

    Returns:
    - X and techniques
    """
    # Specify the data file path
    data_path ='C:\persuasion\static\dataset-individual-Items.csv'
    #data_path = r"D:/persuasion/static/dataset-individual-Items.csv"
    # data_path = '/Users/d18127085/Desktop/statistical_phaseII/static/dataset-individual-Items.csv'

    # Create an instance of the DataSlicer class (Assuming you have defined it)
    data = DataSlicer(data_path)

    # Call the read_data method to read the data from the CSV file
    df = data.read_data()
    
    X = df[
        [
            "age",
            "gender",
            "education",
            "extraversion",
            "agreeableness",
            "conscientiousness",
            "emotional stability",
            "openness",
            "DAS1",
            "DAS2",
            "DAS3",
            "DAS4",
            "DAS5",
            "DAS6",
            "DAS7",
            "DAS8",
            "DAS9",
            "DAS10",
            "DAS11",
            "DAS12",
            "DAS13",
            "DAS14",
            "DAS15",
            "DAS16",
            "DAS17",
            "DAS18",
            "DAS19",
            "DAS20",
            "DAS21",
            "DAS22",
            "DAS23",
            "DAS24",
            "DAS25",
            "DAS26",
            "DAS27",
            "DAS28",
            "DAS29",
            "DAS30",
            "DAS31",
            "DAS32",
            "DAS33",
            "DAS34",
            "DAS35",
        ]
    ]

    df_techniques = df[
        [
            "t10_d1",
            "t10_d2",
            "t10_d3",
            "t9_d1",
            "t9_d2",
            "t9_d3",
            "t8_d1",
            "t8_d2",
            "t8_d3",
            "t7_d1",
            "t7_d2",
            "t7_d3",
            "t6_d1",
            "t6_d2",
            "t6_d3",
            "t5_d1",
            "t5_d2",
            "t5_d3",
            "t4_d1",
            "t4_d2",
            "t4_d3",
            "t3_d1",
            "t3_d2",
            "t3_d3",
            "t2_d1",
            "t2_d2",
            "t2_d3",
            "t1_d1",
            "t1_d2",
            "t1_d3",
        ]
    ]
    print(f"Total features: {X.shape}")
    print(f"Number of techniques: {df_techniques.shape}")

    return X, df_techniques


# Define a function to binarize values based on the criteria
def binarize_value(value):
    if value >= 0 and value <= 5:
        return 0
    elif value >= 6 and value <= 10:
        return 1
    else:
        return None
        

def split_data_df(X, df_techniques, target_technique, test_size=0.33, random_state=42):
    """
    Split the data into training and testing sets for multiple techniques

    Parameters:
    - X: Feature matrix
    - techniques: DataFrame containing target variables
    - target_technique: Name of the column in the DataFrame that contains the technique variable
    - test_size: Fraction of the data to be used as the test set
    - random_state: Random seed for reproducibility

    Returns:
    - X_train, X_test, y_train, y_test: Train and test sets for features and the specified target variable
    """

    y = df_techniques[target_technique]

    # Binarize the target variable
    y = y.apply(binarize_value)

    # Count the occurrences of each unique value in y (class distribution)
    print("Class distribution:")
    print(Counter(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Count the occurrences of each unique value in y_train (class distribution)
    print("Class distribution in y_train:")
    print(Counter(y_train))

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    return X_train, X_test, y_train, y_test


# 
# X, X_train, X_test,y_train, y_test = split_data_df(X, df_techniques, target_technique='t1_d2', test_size=0.33, random_state=42)
