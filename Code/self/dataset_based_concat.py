import os
import pandas as pd

def consolidate_metrics():
    source_base_dir = r"C:\Users\Kai\Documents\Time_Series_Anomaly_Detection\Time-Series-Anomaly-Detection-Seminar\eval\self\metrics\uni"
    target_dataset_based_dir = r"C:\Users\Kai\Documents\Time_Series_Anomaly_Detection\Time-Series-Anomaly-Detection-Seminar\eval\self\dataset_based"

    # Ensure the target directory exists
    os.makedirs(target_dataset_based_dir, exist_ok=True)
    print(f"Target directory for dataset-based metrics: {target_dataset_based_dir}")

    # Iterate through model directories in the source_base_dir
    if not os.path.exists(source_base_dir):
        print(f"Source directory not found: {source_base_dir}")
        return

    for model_name in os.listdir(source_base_dir):
        model_dir_path = os.path.join(source_base_dir, model_name)
        if os.path.isdir(model_dir_path):
            # The model-specific CSV is typically named <model_name>.csv
            model_csv_path = os.path.join(model_dir_path, f"{model_name}.csv")

            if os.path.exists(model_csv_path):
                print(f"Processing model metrics file: {model_csv_path}")
                try:
                    model_df = pd.read_csv(model_csv_path)
                    if 'file' not in model_df.columns:
                        print(f"Skipping {model_csv_path}: 'file' column not found.")
                        continue

                    # Iterate through each row (each row is a dataset's metrics for this model)
                    for index, row in model_df.iterrows():
                        original_dataset_filename = row['file']
                        dataset_file_stem = original_dataset_filename.split('.')[0]
                        
                        target_csv_path = os.path.join(target_dataset_based_dir, f"{dataset_file_stem}.csv")

                        # Prepare the data for the dataset-specific CSV
                        # The new row will have 'Model' as the first column, followed by other metrics
                        data_to_append = row.drop('file').to_dict()
                        data_to_append['Model'] = model_name
                        
                        # Reorder so 'Model' is first, then 'Time', then others
                        metric_columns = [col for col in model_df.columns if col != 'file']
                        output_columns = ['Model'] + metric_columns
                        
                        # Create a DataFrame for the single row with correct column order
                        output_row_df = pd.DataFrame([data_to_append], columns=output_columns)

                        # Check if the target dataset CSV exists to decide on writing header
                        write_header = not os.path.exists(target_csv_path)
                        
                        output_row_df.to_csv(target_csv_path, mode='a', header=write_header, index=False)
                        if write_header:
                            print(f"Created {target_csv_path} and wrote header with data for model {model_name}.")
                        else:
                            print(f"Appended data for model {model_name} to {target_csv_path}.")

                except pd.errors.EmptyDataError:
                    print(f"Skipping empty or invalid CSV: {model_csv_path}")
                except Exception as e:
                    print(f"Error processing file {model_csv_path}: {e}")
            else:
                print(f"Model CSV not found for model {model_name} at {model_csv_path}")
        else:
            print(f"Skipping {model_dir_path} as it is not a directory.")

    print("\nFinished consolidating metrics.")

if __name__ == '__main__':
    consolidate_metrics()