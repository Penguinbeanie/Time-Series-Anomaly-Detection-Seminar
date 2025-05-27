import pandas as pd
import numpy as np
import os
import logging
import time
# Make sure TSB_AD is installed and accessible in your Python environment
try:
    from TSB_AD.evaluation.metrics import get_metrics
    from TSB_AD.utils.slidingWindows import find_length_rank
except ImportError as e:
    print(f"Error importing TSB_AD components: {e}")
    print("Please ensure the TSB_AD library is installed correctly.")
    print("You might need to install it, e.g., pip install TSB-AD")
    exit()

# --- Configuration ---
# REQUIRED: Specify the paths directly here
# Example paths - MODIFY THESE TO MATCH YOUR SYSTEM
SCORE_DIRECTORY_PATH = r'C:\Users\Kai\Documents\Time_Series_Anomaly_Detection\Time-Series-Anomaly-Detection-Seminar\eval\score\uni\AnomalyTransformer'
DATASET_LIST_CSV_PATH = r'C:\Users\Kai\Documents\Time_Series_Anomaly_Detection\Time-Series-Anomaly-Detection-Seminar\TSB-AD\Datasets\File_List\before_crash.csv'
ORIGINAL_DATASET_DIRECTORY_PATH = r'C:\Users\Kai\Documents\Time_Series_Anomaly_Detection\Time-Series-Anomaly-Detection-Seminar\TSB-AD\Datasets\TSB-AD-U'
OUTPUT_METRICS_DIRECTORY_PATH = r'C:\Users\Kai\Documents\Time_Series_Anomaly_Detection\Time-Series-Anomaly-Detection-Seminar\eval\metrics\uni\AnomalyTransformer' # Will save inside this folder
MODEL_NAME = 'AnomalyTransformer' # Crucial: Needs to match the model name used in score filenames and directories

# --- Logging Setup ---
# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_METRICS_DIRECTORY_PATH, exist_ok=True)
LOG_FILE_PATH = os.path.join(OUTPUT_METRICS_DIRECTORY_PATH, f'000_generate_metrics_{MODEL_NAME}.log')

# Configure logging to file and console
# Ensure no duplicate handlers if script is run multiple times in same session (e.g. notebook)
logger = logging.getLogger()
if logger.hasHandlers():
    logger.handlers.clear()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, mode='w'), # 'w' to overwrite log each run
        logging.StreamHandler() # Log to console
    ]
)

# --- Main Processing Function ---
def process_scores_and_generate_metrics():
    """
    Loads scores, original data, calculates metrics, and saves to CSV.
    Uses paths defined in the script's configuration section.
    Dynamically determines metrics from get_metrics output.
    """
    start_time_proc = time.time()
    logging.info("--- Starting Standalone Metrics Generation ---")
    logging.info(f"Model Name: {MODEL_NAME}")
    logging.info(f"Score Directory: {SCORE_DIRECTORY_PATH}")
    logging.info(f"Dataset List File: {DATASET_LIST_CSV_PATH}")
    logging.info(f"Original Dataset Directory: {ORIGINAL_DATASET_DIRECTORY_PATH}")
    logging.info(f"Output Metrics Directory: {OUTPUT_METRICS_DIRECTORY_PATH}")

    # --- Input Validation ---
    if not os.path.isdir(SCORE_DIRECTORY_PATH):
        logging.error(f"Score directory not found: {SCORE_DIRECTORY_PATH}")
        return
    if not os.path.isfile(DATASET_LIST_CSV_PATH):
        logging.error(f"Dataset list file not found: {DATASET_LIST_CSV_PATH}")
        return
    if not os.path.isdir(ORIGINAL_DATASET_DIRECTORY_PATH):
        logging.error(f"Original dataset directory not found: {ORIGINAL_DATASET_DIRECTORY_PATH}")
        return
    # --- End Validation ---

    # --- Read Dataset List ---
    try:
        file_list_df = pd.read_csv(DATASET_LIST_CSV_PATH)
        if 'file_name' not in file_list_df.columns:
            logging.error(f"'file_name' column not found in {DATASET_LIST_CSV_PATH}")
            return
        dataset_filenames = file_list_df['file_name'].values
        logging.info(f"Found {len(dataset_filenames)} dataset entries in {DATASET_LIST_CSV_PATH}")
    except Exception as e:
        logging.error(f"Error reading dataset list file {DATASET_LIST_CSV_PATH}: {e}")
        return
    # --- End Read List ---

    results_list = []
    # Use None as initial state, will be populated by the first successful get_metrics call
    evaluation_result_keys = None

    # --- Process Each File ---
    for base_filename in dataset_filenames:
        filename_prefix = os.path.splitext(base_filename)[0]
        score_file_path = os.path.join(SCORE_DIRECTORY_PATH, f"{MODEL_NAME}_{filename_prefix}.npy")
        original_data_path = os.path.join(ORIGINAL_DATASET_DIRECTORY_PATH, base_filename)

        # 1. Check file existence
        if not os.path.exists(score_file_path):
            logging.warning(f"Score file not found, skipping: {score_file_path}")
            continue
        if not os.path.exists(original_data_path):
            logging.warning(f"Original dataset file not found, skipping: {original_data_path}")
            continue

        logging.info(f"Processing: {base_filename}")

        try:
            # 2. Load original data and labels
            df = pd.read_csv(original_data_path).dropna()
            if df.empty:
                 logging.warning(f"Dataset file is empty after dropna: {original_data_path}. Skipping.")
                 continue
            if 'Label' not in df.columns:
                logging.error(f"'Label' column not found in {original_data_path}. Skipping.")
                continue

            data = df.iloc[:, 0:-1].values.astype(float)
            label = df['Label'].astype(int).to_numpy()

            if data.size == 0 or label.size == 0:
                logging.warning(f"Empty data or label after loading/processing {base_filename}. Skipping.")
                continue

            # 3. Load anomaly scores
            anomaly_scores = np.load(score_file_path)

            # 4. Calculate sliding window
            slidingWindow = 100 # Default fallback
            if data.shape[1] > 0:
                try:
                    slidingWindow = find_length_rank(data[:,0].reshape(-1, 1), rank=1)
                except Exception as slide_e:
                     logging.warning(f"Could not calculate slidingWindow for {base_filename}: {slide_e}. Using default: {slidingWindow}")
            else:
                logging.warning(f"Data for {base_filename} has no columns. Cannot calculate slidingWindow. Using default: {slidingWindow}")


            # 5. Check and adjust score length if necessary
            if len(anomaly_scores) != len(label):
                logging.warning(f"Score length ({len(anomaly_scores)}) and label length ({len(label)}) mismatch for {base_filename}. Adjusting score length.")
                if len(anomaly_scores) > len(label):
                    anomaly_scores = anomaly_scores[:len(label)]
                else:
                    padding_value = anomaly_scores[-1] if len(anomaly_scores) > 0 else 0
                    anomaly_scores = np.pad(anomaly_scores, (0, len(label) - len(anomaly_scores)), 'constant', constant_values=padding_value)
                logging.info(f"Adjusted score length to {len(anomaly_scores)} for {base_filename}.")

            # 6. Calculate metrics
            metric_values_list = None # Use None to indicate failure or success later
            try:
                evaluation_result = get_metrics(anomaly_scores, label, slidingWindow=slidingWindow)
                # Store the keys from the first successful run
                if evaluation_result_keys is None:
                    evaluation_result_keys = list(evaluation_result.keys())
                    logging.info(f"Captured metric keys: {evaluation_result_keys}") # Log this once
                # Ensure the order matches the captured keys for consistency
                metric_values_list = [evaluation_result[key] for key in evaluation_result_keys]
                logging.debug(f'Metrics for {base_filename}: {evaluation_result}')
            except Exception as metrics_e:
                logging.error(f"Error calling get_metrics for {base_filename}: {metrics_e}")
                logging.error(f"Score shape: {anomaly_scores.shape}, Label shape: {label.shape}, Sliding window: {slidingWindow}")
                # metric_values_list remains None

            # 7. Store results: [filename, time_placeholder, list_of_metric_values or None]
            results_list.append([base_filename, 0.0, metric_values_list])

        except pd.errors.EmptyDataError:
             logging.warning(f"Original dataset file is empty or could not be parsed: {original_data_path}. Skipping.")
        except KeyError as e:
             logging.error(f"Column error (e.g., missing 'Label') processing {base_filename}: {e}. Skipping.")
        except Exception as e:
            logging.error(f"An unexpected error occurred processing file {base_filename}: {e}", exc_info=True)

    # --- Save Results to CSV ---
    if not results_list:
        logging.warning("No results were generated. No CSV file will be saved.")
        return

    # Define column headers
    if evaluation_result_keys:
        column_headers = ['file', 'Time'] + evaluation_result_keys
        num_metrics = len(evaluation_result_keys)
    else:
        # Fallback if NO metrics were successfully calculated across ALL files
        logging.error("Failed to calculate metrics for any file. Cannot determine metric names.")
        # Option 1: Save only file and time
        # column_headers = ['file', 'Time']
        # num_metrics = 0
        # Option 2: Use predefined fallback names (less ideal but provides structure)
        fallback_metric_keys = ['AUC_ROC', 'AUC_PR', 'F1', 'Precision', 'Recall', 'Accuracy', 'F1_pa', 'P_AUC', 'VUS_ROC'] # Example, ensure this matches TSB_AD's typical output
        logging.warning(f"Using fallback metric names: {fallback_metric_keys}")
        column_headers = ['file', 'Time'] + fallback_metric_keys
        num_metrics = len(fallback_metric_keys)
        # Set evaluation_result_keys here so NaN padding works correctly below
        evaluation_result_keys = fallback_metric_keys


    # Prepare data for DataFrame, ensuring consistent row length
    final_data_for_df = []
    for row_data in results_list:
        filename, time_placeholder, metric_values = row_data
        if metric_values is None:
            # Metrics failed for this row, pad with NaNs based on the number of expected metrics
             logging.warning(f"Metrics failed for {filename}. Filling with NaN values.")
             metric_padding = [np.nan] * num_metrics
             final_data_for_df.append([filename, time_placeholder] + metric_padding)
        elif len(metric_values) == num_metrics:
            # Metrics succeeded, add the list
            final_data_for_df.append([filename, time_placeholder] + metric_values)
        else:
            # Should ideally not happen if keys are captured correctly, but handle defensively
            logging.error(f"Mismatch in metric count for {filename}. Expected {num_metrics}, got {len(metric_values)}. Padding/Truncating.")
            adjusted_metrics = (metric_values + [np.nan] * num_metrics)[:num_metrics]
            final_data_for_df.append([filename, time_placeholder] + adjusted_metrics)


    if not final_data_for_df:
         logging.error("No valid data rows prepared for saving. Aborting CSV save.")
         return

    try:
        metrics_df = pd.DataFrame(final_data_for_df, columns=column_headers)
        output_csv_path = os.path.join(OUTPUT_METRICS_DIRECTORY_PATH, f"0.2_AnomalyTransformer.csv")
        metrics_df.to_csv(output_csv_path, index=False, float_format='%.5f')
        logging.info(f"Metrics successfully saved to: {output_csv_path}")
    except Exception as e:
        logging.error(f"Failed to save metrics DataFrame to CSV: {e}")

    end_time_proc = time.time()
    logging.info(f"--- Standalone Metrics Generation Finished ---")
    logging.info(f"Total processing time: {end_time_proc - start_time_proc:.2f} seconds")

# --- Run the process ---
if __name__ == "__main__":
    if 'get_metrics' in globals() and 'find_length_rank' in globals():
        process_scores_and_generate_metrics()
    else:
        logging.error("Could not find necessary TSB_AD functions. Aborting.")