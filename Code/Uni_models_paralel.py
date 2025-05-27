import pandas as pd
import numpy as np
import torch
import random, argparse, time, os, logging
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.model_wrapper import *
from TSB_AD.HP_list import Optimal_Uni_algo_HP_dict

# seeding
seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Setup root logger to capture initial prints (optional, but helps fulfill "log all")
# Configure this once before the loop
# Note: This will log initial messages to console. File logging is set up per model inside the loop.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logging.info(f"cuDNN version: {torch.backends.cudnn.version()}")
else:
    logging.info("CUDA not available, cuDNN version not applicable.")


if __name__ == '__main__':

    Start_T = time.time()
    ## ArgumentParser
    model_list = ['Left_STAMPi']
    
    # 'KMeansAD_U','AutoEncoder', 'CNN', 'LSTMAD', 'TranAD', 'USAD', 'OmniAnomaly', 'FITS', 'M2N2']
    # ['MOMENT_ZS', 'FFT', 'SR', 'Sub_IForest', 'IForest', 'LOF', 'Sub_LOF', 'POLY', 'MatrixProfile', 'Sub_PCA',
                #'Sub_HBOS', 'Sub_KNN', 

    for model in model_list:
        # Reset file handler for each model to log to a model-specific file
        # Remove existing handlers to avoid duplicate logging or logging to wrong files
        for handler in logging.root.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                logging.root.removeHandler(handler)
                handler.close() # Close the file handler explicitly

        try:
            ## ArgumentParser - Setup defaults for this model iteration
            # Using parse_args([]) ensures defaults are used when no command-line args are given
            parser = argparse.ArgumentParser(description='Generating Anomaly Score')
            parser.add_argument('--dataset_dir', type=str, default=r'C:\Users\Kai\Documents\Time_Series_Anomaly_Detection\Time-Series-Anomaly-Detection-Seminar\TSB-AD\Datasets\TSB-AD-U')
            parser.add_argument('--file_list', type=str, default=r'C:\Users\Kai\Documents\Time_Series_Anomaly_Detection\Time-Series-Anomaly-Detection-Seminar\TSB-AD\Datasets\File_List\TSB-AD-U-Eva.csv')
            parser.add_argument('--score_dir', type=str, default='eval/score/uni/')
            parser.add_argument('--save_dir', type=str, default='eval/metrics/uni/')
            parser.add_argument('--no-save', action='store_false', dest='save', default=True, help='Disable saving metrics') # Note: store_false means the flag disables saving
            parser.add_argument('--AD_Name', type=str, default=model) # Set current model as default

            args = parser.parse_args([]) # Parse an empty list to use defaults

            # Create directories for scores and metrics for the current model
            target_dir = os.path.join(args.score_dir, args.AD_Name)
            target_dir_metrics = os.path.join(args.save_dir, args.AD_Name)
            os.makedirs(target_dir, exist_ok=True)
            os.makedirs(target_dir_metrics, exist_ok=True)

            # Configure file logging specifically for this model run
            log_file_path = os.path.join(target_dir, f'000_run_{args.AD_Name}.log')
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logging.getLogger().addHandler(file_handler) # Add handler to the root logger

            logging.info(f"--- Starting processing for model: {args.AD_Name} ---")
            logging.info(f"Using arguments: {args}")

            file_list = pd.read_csv(args.file_list)['file_name'].values
            
            # Check if the model name exists in the dictionary before accessing it
            if args.AD_Name not in Optimal_Uni_algo_HP_dict:
                logging.error(f"Optimal hyperparameters not found for model: {args.AD_Name}. Skipping this model.")
                continue # Skip to the next model in the outer loop
                
            Optimal_Det_HP = Optimal_Uni_algo_HP_dict[args.AD_Name]
            logging.info(f'Optimal_Det_HP: {Optimal_Det_HP}')

            evaluation_result = {} # Initialize evaluation_result dict to ensure it exists even if saving is off or loop is skipped

            metrics_csv_path = os.path.join(target_dir_metrics, f"{args.AD_Name}.csv")
            # Check if the CSV already exists to determine if header should be written
            first_write = not os.path.exists(metrics_csv_path)
            # Initialize column names holder
            col_w = None

            for filename in file_list:
                score_file_path = os.path.join(target_dir, f"{args.AD_Name}_{filename.split('.')[0]}.npy")
                if os.path.exists(score_file_path):
                    logging.info(f"Score file already exists, skipping: {score_file_path}")
                    continue

                logging.info(f'Processing: {filename} by {args.AD_Name}')

                file_path = os.path.join(args.dataset_dir, filename)
                try:
                    df = pd.read_csv(file_path).dropna()
                    data = df.iloc[:, 0:-1].values.astype(float)
                    label = df['Label'].astype(int).to_numpy()
                    # logging.info(f'data shape: {data.shape}') # Optional: log shape
                    # logging.info(f'label shape: {label.shape}') # Optional: log shape

                    feats = data.shape[1]
                    slidingWindow = find_length_rank(data[:,0].reshape(-1, 1), rank=1)
                    # Ensure train_index calculation is robust if filename format changes
                    try:
                        train_index_str = filename.split('.')[0].split('_')[-3]
                        train_index = int(train_index_str)
                        if train_index > len(data) or train_index < 0:
                           raise ValueError(f"Invalid train_index {train_index} for data length {len(data)}")
                        data_train = data[:train_index, :]
                    except (IndexError, ValueError) as e:
                        logging.error(f"Could not determine train_index from filename {filename}: {e}. Skipping file.")
                        continue # Skip this file

                    start_time = time.time()

                    # Make sure the Pools are imported correctly
                    if args.AD_Name in Semisupervise_AD_Pool:
                        output = run_Semisupervise_AD(args.AD_Name, data_train, data, **Optimal_Det_HP)
                    elif args.AD_Name in Unsupervise_AD_Pool:
                        output = run_Unsupervise_AD(args.AD_Name, data, **Optimal_Det_HP)
                    else:
                        
                        logging.error(f"{args.AD_Name} is not defined in known supervised/unsupervised pools.")
                        raise Exception(f"{args.AD_Name} is not defined")

                    end_time = time.time()
                    run_time = end_time - start_time

                    if isinstance(output, np.ndarray):
                        logging.info(f'Success at {filename} using {args.AD_Name} | Time cost: {run_time:.3f}s at length {len(label)}')
                        np.save(score_file_path, output) # Use the defined score_file_path
                    else:
                        # If 'output' contains an error message (string)
                        logging.error(f'At {filename}: {output}')
                        # Skip metrics calculation if model run failed
                        continue

                    ### whether to save the evaluation result
                    if args.save:
                        
                        logging.info("Calculating metrics and appending to CSV.")
                        try:
                            # Handle potential length mismatch.
                            if len(output) != len(label):
                                logging.warning(f"Output length ({len(output)}) and label length ({len(label)}) mismatch for {filename}. Padding or truncating output.")
                                if len(output) > len(label):
                                    output = output[:len(label)]
                                else:
                                    padding_value = output[-1] if len(output) > 0 else 0
                                    output = np.pad(output, (0, len(label) - len(output)), 'constant', constant_values=padding_value)
                                logging.info(f"Adjusted output length to {len(output)}.")


                            evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow)
                            logging.info(f'evaluation_result: {evaluation_result}') # Keep logging result


                            # Define column names ONCE using the first successful result
                            if col_w is None:
                                if evaluation_result: # Check if get_metrics was successful
                                     col_w = list(evaluation_result.keys())
                                     col_w.insert(0, 'Time')
                                     col_w.insert(0, 'file')
                                else:
                                     logging.error("Cannot determine column names because evaluation_result is empty.")
                                     # Handle error - maybe raise or set default columns?
                                     # For now, we'll skip appending if columns can't be determined
                                     continue # Skip appending for this file


                            # Check if col_w was successfully set before proceeding
                            if col_w is None:
                                logging.error(f"Column names not set, cannot append metrics for {filename}.")
                                continue


                            # Get values and prepare row data
                            list_w = list(evaluation_result.values()) # Get values from the current result
                            list_w.insert(0, run_time)
                            list_w.insert(0, filename)

                           
                           

                            # Create a DataFrame for the single row
                            row_df = pd.DataFrame([list_w], columns=col_w)

                            #
                            # Use mode='a' for append, write header only if it's the first write
                            write_header = first_write
                            row_df.to_csv(metrics_csv_path, mode='a', header=write_header, index=False)

                            
                            first_write = False

                            
                            logging.info(f"Appended metrics for {filename} to {metrics_csv_path}")

                        except Exception as e:
                           
                            logging.error(f"Error calculating metrics or appending to CSV for {filename}: {e}")
                            logging.error(f"Output shape: {output.shape if isinstance(output, np.ndarray) else 'N/A'}, Label shape: {label.shape}, Sliding window: {slidingWindow}")
                            

                except FileNotFoundError:
                    logging.error(f"Dataset file not found: {file_path}")
                except Exception as e:
                    logging.error(f"Error processing file {filename} with model {args.AD_Name}: {e}")
                    # Optionally continue to the next file or break/reraise depending on desired behavior


        except ImportError as e:
             logging.error(f"Import error for model {model}: {e}. Check TSB_AD installation and dependencies.")
             # Continue to next model if one fails due to missing optional dependencies perhaps
        except KeyError as e:
             logging.error(f"KeyError encountered for model {model}: {e}. This might indicate the model name is missing from HP_list or Pools.")
        except Exception as e:
            # Log any other exceptions during the setup or processing for a specific model
            logging.error(f"An unexpected error occurred while processing model {model}: {e}", exc_info=True) # exc_info logs traceback

        finally:
            # Ensure the file handler added for this model is removed and closed
            for handler in logging.root.handlers[:]:
                if handler is file_handler: # Check specifically for the handler added in this iteration
                    logging.root.removeHandler(handler)
                    handler.close()
                    break # Found and removed the handler
            logging.info(f"--- Finished processing for model: {model} ---")


    End_T = time.time()
    logging.info(f"\n\nTotal Running Time: {(End_T - Start_T) / 60:.2f} min")