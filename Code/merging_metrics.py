import pandas as pd
from pathlib import Path
import os

# --- Configuration ---
# Define the source directory containing the model subfolders
source_dir_path = Path(r"C:\Users\Kai\Documents\Time_Series_Anomaly_Detection\Time-Series-Anomaly-Detection-Seminar\eval\metrics\multi")

# Define the destination directory for the merged file
dest_dir_path = Path(r"C:\Users\Kai\Documents\Time_Series_Anomaly_Detection\Time-Series-Anomaly-Detection-Seminar\eval\metrics\multi\all")

# Define the name for the output merged CSV file
output_file_name = "merged_all_multi_metrics.csv"

# Define the expected columns in the source CSVs (for potential validation, though not strictly used here)
expected_columns = ['file','Time','AUC-PR','AUC-ROC','VUS-PR','VUS-ROC','Standard-F1','PA-F1','Event-based-F1','R-based-F1','Affiliation-F']

# Define the desired order of columns in the final output file
output_columns = ['model','file','Time','AUC-PR','AUC-ROC','VUS-PR','VUS-ROC','Standard-F1','PA-F1','Event-based-F1','R-based-F1','Affiliation-F']
# --- End Configuration ---

def merge_csvs(source_dir: Path, dest_dir: Path, output_filename: str, final_columns: list):
    """
    Merges CSV files from unique subdirectories within a source directory
    into a single CSV file in the destination directory, adding a 'model' column.

    Args:
        source_dir (Path): The path to the directory containing subfolders.
        dest_dir (Path): The path to the directory where the merged file will be saved.
        output_filename (str): The name for the merged output CSV file.
        final_columns (list): The desired list and order of columns for the output file.
    """
    all_dataframes = []
    models_processed = 0
    files_found = 0

    if not source_dir.is_dir():
        print(f"Error: Source directory not found: {source_dir}")
        return

    print(f"Scanning source directory: {source_dir}")

    # Iterate through each item in the source directory
    for item in source_dir.iterdir():
        # Check if the item is a directory (a model folder)
        if item.is_dir():
            model_name = item.name  # Get the subdirectory name (model name)
            print(f"  Processing model directory: {model_name}")

            csv_files_in_subdir = list(item.glob('*.csv'))

            if not csv_files_in_subdir:
                print(f"    Warning: No CSV file found in directory: {item}. Skipping.")
                continue
            elif len(csv_files_in_subdir) > 1:
                print(f"    Warning: Multiple CSV files found in directory: {item}. Using the first one: {csv_files_in_subdir[0].name}")

            # Assume the first CSV found is the correct one
            csv_file_path = csv_files_in_subdir[0]
            files_found += 1

            try:
                # Read the CSV file into a pandas DataFrame
                df = pd.read_csv(csv_file_path)

                # Add the 'model' column at the beginning
                df.insert(0, 'model', model_name)

                # Append the dataframe to the list
                all_dataframes.append(df)
                models_processed += 1
                print(f"    Successfully read and processed: {csv_file_path.name}")

            except pd.errors.EmptyDataError:
                print(f"    Warning: CSV file is empty: {csv_file_path}. Skipping.")
            except Exception as e:
                print(f"    Error reading or processing file {csv_file_path}: {e}")

    # Check if any dataframes were collected
    if not all_dataframes:
        print("\nNo valid CSV files found or processed. No output file generated.")
        return

    print(f"\nFound {files_found} CSV files in {models_processed} model directories.")

    # Concatenate all dataframes into a single one
    print("Concatenating all dataframes...")
    merged_df = pd.concat(all_dataframes, ignore_index=True)

    # Ensure the columns are in the desired order
    # Check if all expected columns exist, handle missing ones if necessary (optional)
    # For now, we assume all columns exist and just reorder
    try:
        merged_df = merged_df[final_columns]
    except KeyError as e:
        print(f"\nWarning: Could not reorder columns. Missing columns: {e}")
        print("The output file will contain columns as read, with 'model' added first.")
        # Fallback: keep the order as is after adding 'model'
        pass # merged_df already has 'model' first from insert

    # Create the destination directory if it doesn't exist
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Ensured destination directory exists: {dest_dir}")

    # Define the full path for the output file
    output_file_path = dest_dir / output_filename

    # Save the merged dataframe to a new CSV file
    try:
        merged_df.to_csv(output_file_path, index=False, encoding='utf-8')
        print(f"\nSuccessfully merged {len(merged_df)} rows into: {output_file_path}")
    except Exception as e:
        print(f"\nError writing output file {output_file_path}: {e}")

# --- Run the merging process ---
if __name__ == "__main__":
    merge_csvs(source_dir_path, dest_dir_path, output_file_name, output_columns)