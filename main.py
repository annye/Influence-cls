


import argparse
from dataloading import *
from pdb import set_trace
from shap_rfa_rfe import *


def main():
    """Main method."""
    parser = argparse.ArgumentParser(description="Script accepts a filepath string and a list of strings (ML algos) as arguments.")
    #parser.add_argument("Input_File_Path", nargs='?', default="check_string_for_empty",
    #                     help="A string argument denoting the filepath of the input data.")
    #parser.add_argument("Results_File_Path", nargs='?', default="check_string_for_empty",
    #                     help="A string argument denoting the filepath of the results data.")
    parser.add_argument('ML_List', type=str, help="A comma-separated list of strings denoting the ML algos you wish to compare.")
    args = parser.parse_args()
    
    #algos = args.ML_List.split(',')

    # Read in input data
    X, df_techniques = get_data()
    #X = X[0:100]
    # Iterate over the techniques
    for col in df_techniques.columns:
        # Get train/test splits
        y = df_techniques[col]
        y = y.apply(binarize_value)
        #y = y[0:100]

        build_models(X, y, args.ML_List, col)

     
    #best_algo = choose_ML_algorithm(input_data, args.Results_File_Path, algos)
    

if __name__ == "__main__":
    main()
