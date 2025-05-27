import numpy as np
import os

def convert_npy_to_csv(npy_file_path):
    """
    Converts a .npy file to a .csv file.

    The .csv file will be saved in the same directory and with the same
    base name as the .npy file, but with a .csv extension.

    Args:
        npy_file_path (str): The path to the .npy file.
    """
    if not os.path.exists(npy_file_path):
        print(f"Error: File not found at {npy_file_path}")
        return

    if not npy_file_path.endswith('.npy'):
        print(f"Error: Input file '{npy_file_path}' is not a .npy file.")
        return

    base_name = os.path.splitext(npy_file_path)[0]
    csv_file_path = base_name + ".csv"

    try:
        # Load the data from the .npy file
        data = np.load(npy_file_path)

        # If the data is 1D, reshape it to be a column vector for CSV
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        # Save the data to a .csv file
        # Using a comma as a delimiter, and a common format for floats
        np.savetxt(csv_file_path, data, delimiter=",", fmt='%f')

        print(f"Successfully converted '{npy_file_path}' to '{csv_file_path}'")

    except Exception as e:
        print(f"An error occurred during conversion: {e}")

if __name__ == "__main__":
    # Get the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the .npy file relative to the script's location
    # This assumes the .npy file is in the specified path relative to the project root
    # If the script is in the project root, this path works.
    # Otherwise, adjust npy_file_to_convert accordingly.
    project_root = os.path.dirname(script_dir) # Assuming script is one level down, e.g. in a 'scripts' folder
                                            # If script is at root, project_root = script_dir

    # --- IMPORTANT: SET THE CORRECT PATH TO YOUR .NPY FILE HERE ---
    # Example using the path from your attached file, assuming the script is in the project root:
    # npy_file_to_convert = "eval/self/score/uni/POLY/POLY_001_CPU1_id_1_Hardware_tr_1150_1st_1196.npy"

    # If you want to make it more dynamic, you could use argparse to pass the file path as an argument.
    # For now, we'll use the specific file path from your example.
    # Please ensure this path is correct from where you run the script.
    
    # Path to the .npy file based on the user's attached file
    # Assuming the script 'npy_to_csv_converter.py' is at the root of the workspace
    # /c%3A/Users/Kai/Documents/Time_Series_Anomaly_Detection/Time-Series-Anomaly-Detection-Seminar/
    target_npy_file = "eval/self/predictions/uni/SAND/PRED_SAND_003_CPU1_id_03_Hardware_tr_1169_1st_1172.npy"
    
    # Construct the absolute path if needed, or ensure the relative path is correct
    # For simplicity, using the relative path directly as it's common for scripts within a project
    
    convert_npy_to_csv(target_npy_file)