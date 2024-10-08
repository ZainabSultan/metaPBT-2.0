import os
import json
import pandas as pd

# def extract_identifier(subdir_name):
#     """
#     Extracts the unique identifier (the number between the third underscore and the fourth underscore).
#     Example: For pbt_function_75563_00000_0_2024-09-12_13-52-16, it returns '00000_0'.
#     """
#     parts = subdir_name.split('_')
#     if len(parts) >= 5:
#         return f"{parts[3]}_{parts[4]}"
#     return None

def extract_cur_lr(file_path, name='cur_lr'):
    """Extracts the first value of the 'cur_lr' column from a CSV file."""
    df = pd.read_csv(file_path)
    if name in df.columns:
        return df[name].iloc[0]  # Get the first value in the 'cur_lr' column
    return None

def compare_csv(file1, file2, column_names=['mean_accuracy', 'cur_lr', 'optimal_lr', 'q_err', 'done', 'training_iteration']):
    """Compares specific columns of two CSV files by column names using pandas."""
    df1 = pd.read_csv(file1, usecols=column_names)
    df2 = pd.read_csv(file2, usecols=column_names)
    return df1.equals(df2)

def load_json_file(file_path):
    """
    Loads a JSON file where each line is a separate JSON object and returns it as a list of dictionaries.
    """
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def compare_json(file1, file2):
    """
    Compares two JSON files where each line is a separate JSON object by loading the full content into memory.
    """
    json_data1 = load_json_file(file1)
    json_data2 = load_json_file(file2)
    return json_data1 == json_data2

def find_counterparts(dir1, dir2, name='cur_lr'):
    """Finds and matches counterpart CSV files between two directories based on 'cur_lr'."""
    lr_to_file1 = {}
    lr_to_file2 = {}

    # Read all CSV files in dir1
    for sample in dir1:
        for filename in os.listdir(sample):
            if filename.endswith('.csv'):
                file_path = os.path.join(sample, filename)
                cur_lr = extract_cur_lr(file_path, name=name)
                print(cur_lr)
                if cur_lr is not None:
                    lr_to_file1[cur_lr] = file_path

    # Read all CSV files in dir2
    for sample in dir2:
        for filename in os.listdir(sample):
            if filename.endswith('.csv'):
                file_path = os.path.join(sample, filename)
                cur_lr = extract_cur_lr(file_path, name=name)
                print(cur_lr)
                if cur_lr is not None:
                    lr_to_file2[cur_lr] = file_path

    # Find matches based on 'cur_lr'
    matches = {}
    for lr, file1_path in lr_to_file1.items():
        if lr in lr_to_file2:
            file2_path = lr_to_file2[lr]
            matches[file1_path] = file2_path
    print(matches)
    return matches

def compare_files_in_subdirs(dir1, dir2):
    """
    Loops through the subdirectories of the given main subdirs in dir1 and dir2,
    compares progress.csv and result.json based on the identifier and prints whether they are identical.
    """
    # Find the subdirs within 'x' in both dir1 and dir2
    subdir_x_dir1 = os.path.join(dir1, os.listdir(dir1)[1])  # Assuming 'x' is the only subdir in dir1
    subdir_x_dir2 = os.path.join(dir2, os.listdir(dir2)[1])  # Assuming 'x' is the only subdir in dir2

    subdirs_dir1 = [os.path.join(subdir_x_dir1, d) for d in os.listdir(subdir_x_dir1) if os.path.isdir(os.path.join(subdir_x_dir1, d))]
    subdirs_dir2 = [os.path.join(subdir_x_dir2, d) for d in os.listdir(subdir_x_dir2) if os.path.isdir(os.path.join(subdir_x_dir2, d))]

    matches = find_counterparts(subdirs_dir1, subdirs_dir2, name='info/learner/default_policy/learner_stats/cur_lr')
    
    # Compare the matched files
    for file1, file2 in matches.items():
        subdir1 = os.path.basename(os.path.dirname(file1))
        subdir2 = os.path.basename(os.path.dirname(file2))

        # Compare progress.csv files
        progress_file_dir1 = file1
        progress_file_dir2 = file2
        print(f"Comparing progress.csv files in {subdir1} and {subdir2}:")
        
        if os.path.exists(progress_file_dir1) and os.path.exists(progress_file_dir2):
            are_csv_identical = compare_csv(progress_file_dir1, progress_file_dir2, column_names=['info/learner/default_policy/learner_stats/cur_lr'])
        else:
            are_csv_identical = False
        
        # Compare result.json files
        result_file_dir1 = os.path.join(os.path.dirname(file1), 'result.json')
        result_file_dir2 = os.path.join(os.path.dirname(file2), 'result.json')

        if os.path.exists(result_file_dir1) and os.path.exists(result_file_dir2):
            are_json_identical = compare_json(result_file_dir1, result_file_dir2)
        else:
            are_json_identical = False

        print(f" - CSV files identical: {are_csv_identical}")
        print(f" - JSON files identical: {are_json_identical}")
        print()

# Example usage
dir1 = '/home/fr/fr_fr/fr_zs53/DKL/metaPBT-2.0/testing_dir/2024-09-19_18:46:37_PPO_gravity_1.0_metadkl_Size4_CARLCartPole_timesteps_total'#'/home/fr/fr_fr/fr_zs53/DKL/metaPBT-2.0/testing_dir/20240912_134948_0_pb2_dkl_Size_4_repro_test_toy_func'
dir2 = '/home/fr/fr_fr/fr_zs53/DKL/metaPBT-2.0/testing_dir/2024-09-12_20:58:29_PPO_length_0.05_pb2_Size4_CARLCartPole_timesteps_total'#'/home/fr/fr_fr/fr_zs53/DKL/metaPBT-2.0/testing_dir/20240912_135204_0_pb2_dkl_Size_4_repro_test_toy_func'

compare_files_in_subdirs(dir1, dir2)
