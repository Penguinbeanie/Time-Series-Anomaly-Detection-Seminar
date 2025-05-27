import pandas as pd
import numpy as np
import torch
import random, time, os, logging
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.model_wrapper import *
from TSB_AD.HP_list import Optimal_Multi_algo_HP_dict
import multiprocessing

# seeding
def set_seed(seed_value):

    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True




def run_models(list_of_models_to_run, dataset_dir, file_list_path, score_dir_base, metrics_dir_base, save_metrics, hp_dict, seed_value, process_id):

    # starting timer
    process_start_time = time.time()

    # starting logging
    logging.info(f"--- Process {process_id} (PID: {os.getpid()}) started, handling models: {list_of_models_to_run} ---")

    # laoding file_list
    try:
        file_list_df = pd.read_csv(file_list_path)
        file_list = file_list_df['file_name'].values
    except FileNotFoundError:
        logging.error(f"[Process {process_id}] File list not found at {file_list_path}. Exiting process.")
        return
    except Exception as e:
        logging.error(f"[Process {process_id}] Error loading file list: {e}. Exiting process.")
        return
    

    # starting model loop
    for model_name in list_of_models_to_run:
        model_start_time = time.time()
        file_handler = None 
        logger = logging.getLogger()

        try:
            # Create directories for scores and metrics for the current model
            target_dir = os.path.join(score_dir_base, model_name)
            target_dir_metrics = os.path.join(metrics_dir_base, model_name)
            os.makedirs(target_dir, exist_ok=True)
            os.makedirs(target_dir_metrics, exist_ok=True)

            # Configure file logging specifically for this model run
            log_file_path = os.path.join(target_dir, f'000_run_{model_name}.log')
            file_handler = logging.FileHandler(log_file_path, mode='a') # Append mode
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(process)d - %(levelname)s - %(message)s'))
            logger.addHandler(file_handler) # Add handler to the root logger for this model

            logger.info(f"--- [Model: {model_name}] Starting processing (Process {process_id}) ---")
            logger.info(f"[Model: {model_name}] Using arguments: dataset_dir={dataset_dir}, file_list_path={file_list_path}, score_dir_base={score_dir_base}, metrics_dir_base={metrics_dir_base}, save_metrics={save_metrics}, seed_value={seed_value}")
            
            # Check if the model name exists in the dictionary before accessing it
            if model_name not in hp_dict:
                logger.error(f"[Model: {model_name}] Optimal hyperparameters not found. Skipping this model.")
                if file_handler:
                    logger.removeHandler(file_handler)
                    file_handler.close()
                    file_handler = None
                continue # Skip to the next model
                
            Optimal_Det_HP = hp_dict[model_name]
            logger.info(f'[Model: {model_name}] Optimal_Det_HP: {Optimal_Det_HP}')

            metrics_csv_path = os.path.join(target_dir_metrics, f"{model_name}.csv")
            # Check if the CSV already exists to determine if header should be written
            first_write = not os.path.exists(metrics_csv_path)
            # Initialize column names holder
            col_w = None

            for filename in file_list:
                score_file_path = os.path.join(target_dir, f"{model_name}_{filename.split('.')[0]}.npy")
                if os.path.exists(score_file_path):
                    logger.info(f"[Model: {model_name}] Score file already exists, skipping: {score_file_path}")
                    continue

                logger.info(f'[Model: {model_name}] Processing: {filename}')

                file_path = os.path.join(dataset_dir, filename)
                try:
                    df = pd.read_csv(file_path).dropna()
                    data = df.iloc[:, 0:-1].values.astype(float)
                    label = df['Label'].astype(int).to_numpy()

                    slidingWindow = find_length_rank(data[:,0].reshape(-1, 1), rank=1)
                    try:
                        train_index_str = filename.split('.')[0].split('_')[-3]
                        train_index = int(train_index_str)
                        if train_index > len(data) or train_index < 0:
                           raise ValueError(f"Invalid train_index {train_index} for data length {len(data)}")
                        data_train = data[:train_index, :]
                    except (IndexError, ValueError) as e:
                        logger.error(f"[Model: {model_name}] Could not determine train_index from filename {filename}: {e}. Skipping file.")
                        continue

                    file_proc_start_time = time.time()

                    output = None
                    if model_name in Semisupervise_AD_Pool:
                        output = run_Semisupervise_AD(model_name, data_train, data, **Optimal_Det_HP)
                    elif model_name in Unsupervise_AD_Pool:
                        output = run_Unsupervise_AD(model_name, data, **Optimal_Det_HP)
                    else:
                        logger.error(f"[Model: {model_name}] {model_name} is not defined in known supervised/unsupervised pools.")
                        continue

                    file_proc_end_time = time.time()
                    run_time = file_proc_end_time - file_proc_start_time

                    if isinstance(output, np.ndarray):
                        logger.info(f'[Model: {model_name}] Success at {filename} | Time cost: {run_time:.3f}s at length {len(label)}')
                        np.save(score_file_path, output)
                    else:
                        logger.error(f'[Model: {model_name}] At {filename}: {output}')
                        continue

                    if save_metrics:
                        try:
                            if len(output) != len(label):
                                logger.warning(f"[Model: {model_name}] Output length ({len(output)}) and label length ({len(label)}) mismatch for {filename}. Padding or truncating output.")
                                if len(output) > len(label):
                                    output = output[:len(label)]
                                else:
                                    padding_value = output[-1] if len(output) > 0 else 0
                                    output = np.pad(output, (0, len(label) - len(output)), 'constant', constant_values=padding_value)
                                logger.info(f"[Model: {model_name}] Adjusted output length to {len(output)}.")

                            evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow)

                            if col_w is None:
                                if evaluation_result:
                                     col_w = list(evaluation_result.keys())
                                     col_w.insert(0, 'Time')
                                     col_w.insert(0, 'file')
                                else:
                                     logger.error(f"[Model: {model_name}] Cannot determine column names because evaluation_result is empty for {filename}.")
                                     continue

                            if col_w is None:
                                logger.error(f"[Model: {model_name}] Column names not set, cannot append metrics for {filename}.")
                                continue

                            list_w = list(evaluation_result.values())
                            list_w.insert(0, run_time)
                            list_w.insert(0, filename)

                            row_df = pd.DataFrame([list_w], columns=col_w)
                            write_header = first_write
                            row_df.to_csv(metrics_csv_path, mode='a', header=write_header, index=False)
                            first_write = False

                        except Exception as e:
                            logger.error(f"[Model: {model_name}] Error calculating metrics or appending to CSV for {filename}: {e}", exc_info=True)
                            logger.error(f"[Model: {model_name}] Output shape: {output.shape if isinstance(output, np.ndarray) else 'N/A'}, Label shape: {label.shape}, Sliding window: {slidingWindow}")

                except FileNotFoundError:
                    logger.error(f"[Model: {model_name}] Dataset file not found: {file_path}")
                except Exception as e:
                    logger.error(f"[Model: {model_name}] Error processing file {filename}: {e}", exc_info=True)
            # --- End of file loop ---

        except ImportError as e:
             logger.error(f"[Model: {model_name}] Import error: {e}. Check TSB_AD installation/dependencies.")
        except KeyError as e:
             logger.error(f"[Model: {model_name}] KeyError: {e}. Missing from HP dict?")
        except Exception as e:
            logger.error(f"[Model: {model_name}] An unexpected error occurred: {e}", exc_info=True)
        finally:
            if file_handler:
                logger.removeHandler(file_handler)
                file_handler.close()
                file_handler = None
            model_end_time = time.time()
            logging.info(f"--- [Model: {model_name}] Finished processing (Process {process_id}). Duration: {(model_end_time - model_start_time):.2f}s ---")
    # --- End of model loop ---

    process_end_time = time.time()
    logging.info(f"--- Process {process_id} (PID: {os.getpid()}) finished. Total Duration: {(process_end_time - process_start_time)/60:.2f} min ---")



if __name__ == '__main__':
    # --- Configuration ---
    SEED = 2024
    DATASET_DIR = r'C:\Users\Kai\Documents\Time_Series_Anomaly_Detection\Time-Series-Anomaly-Detection-Seminar\TSB-AD\Datasets\TSB-AD-M'
    FILE_LIST_PATH = r'C:\Users\Kai\Documents\Time_Series_Anomaly_Detection\Time-Series-Anomaly-Detection-Seminar\TSB-AD\Datasets\File_List\TSB-AD-M-Eva.csv'
    SCORE_DIR_BASE = 'eval/score/multi/'
    METRICS_DIR_BASE = 'eval/metrics/multi/'
    SAVE_METRICS = True

    # model lists for each process
    model_list_1 = ['HBOS', 'OCSVM', 'MCD', 'KNN'] #DONE: 'IForest', 'LOF', PROBLEM:  'PCA'
    model_list_2 = ['RobustPCA'] #DONE: 'KMeansAD', 'KShapeAD', 'COPOD', 'CBLOF', PROBLEM:  'EIF'
    model_list_3 = ['AutoEncoder', 'CNN', 'LSTMAD', 'TranAD', 'OmniAnomaly', 'USAD', 'FITS'] #DONE: 'AnomalyTransformer', 

    
    list_of_model_lists = [model_list_1, model_list_3]

    NUM_PROCESSES = len(list_of_model_lists)

    # --- Global Setup ---
    Global_Start_T = time.time() # Define Start time HERE

    set_seed(SEED)

    # console logging for the main process
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info(f"--- Main Process Started (PID: {os.getpid()}) ---")
    logging.info(f"Launching {NUM_PROCESSES} separate processes based on manual lists.")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"cuDNN version: {torch.backends.cudnn.version()}")
    else:
        logging.info("CUDA not available, cuDNN version not applicable.")

    # check dependencies
    try:
        # Make sure these Pools are relevant for multivariate models or adjust check
        if 'Semisupervise_AD_Pool' not in globals() or 'Unsupervise_AD_Pool' not in globals():
             raise NameError("AD Pools (Semisupervise_AD_Pool, Unsupervise_AD_Pool) are not defined/imported globally.")
        if 'Optimal_Multi_algo_HP_dict' not in globals():
             raise NameError("Optimal_Multi_algo_HP_dict is not defined/imported globally.")
    except NameError as e:
        logging.error(f"Setup Error: {e}. Ensure necessary variables/imports are available globally. Exiting.")
        exit()

    processes = []

    # launch Processes
    logging.info("Launching processes...")

    for i, current_model_list in enumerate(list_of_model_lists): # Use enumerate
        process_id = i + 1
        if not current_model_list:
            logging.warning(f"Skipping process {process_id} as its model list is empty.")
            continue
        logging.info(f"Assigning models to process {process_id}: {current_model_list}")

        # arguments for run_models
        args_for_process = (
            current_model_list,
            DATASET_DIR,
            FILE_LIST_PATH,
            SCORE_DIR_BASE,
            METRICS_DIR_BASE,
            SAVE_METRICS,
            Optimal_Multi_algo_HP_dict, # Make sure this is the correct HP dict
            SEED,
            process_id
        )

        p = multiprocessing.Process(target=run_models, args=args_for_process)
        processes.append(p)
        p.start()

    # Wait for Processes to Complete 
    if not processes:
        logging.warning("No processes were started (all model lists might be empty).")
    else:
        logging.info(f"Waiting for {len(processes)} processes to finish...")
        for i, p in enumerate(processes):
            # Determine original ID based on the index in the 'processes' list
            # This assumes processes list only contains started processes
            original_process_id = i + 1 # Simple mapping if no skips occurred
            p.join()
            logging.info(f"Process {original_process_id} finished.")

    # Finalization
    Global_End_T = time.time() # Define End time HERE
    logging.info(f"--- All Launched Processes Finished ---")
    logging.info(f"\n\nTotal Global Running Time: {(Global_End_T - Global_Start_T) / 60:.2f} min") # Now calculation is safe
