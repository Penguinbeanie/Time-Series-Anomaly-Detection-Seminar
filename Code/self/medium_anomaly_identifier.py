import csv

def identify_anomalies(data_values, min_val=530, max_val=700, min_len=3, max_len=4):
    """
    Identifies anomalies in a list of data values.
    Anomalies are sequences of exactly 3 or 4 consecutive points > min_val and < max_val.
    """
    num_points = len(data_values)
    predicted_anomalies = [0] * num_points
    
    i = 0
    while i < num_points:
        # Check if current point is in the target range
        if data_values[i] > min_val and data_values[i] < max_val:
            # Found start of a potential sequence - count consecutive points in range
            start_index = i
            sequence_length = 0
            
            # Count all consecutive points in the range
            while i < num_points and data_values[i] > min_val and data_values[i] < max_val:
                sequence_length += 1
                i += 1
            
            # Check if sequence length is exactly 3 or 4
            if sequence_length == 3 or sequence_length == 4:
                # Mark all points in this sequence as anomalies
                for j in range(start_index, start_index + sequence_length):
                    predicted_anomalies[j] = 1
            
            # i is already positioned at the next point after the sequence
        else:
            # Point is not in range, move to next point
            i += 1
    
    return predicted_anomalies

def calculate_metrics(true_labels, predicted_labels):
    """
    Calculates True Positives, False Positives, and False Negatives.
    """
    tp = 0
    fp = 0
    fn = 0
    
    for true, pred in zip(true_labels, predicted_labels):
        if pred == 1 and true == 1:
            tp += 1
        elif pred == 1 and true == 0:
            fp += 1
        elif pred == 0 and true == 1:
            fn += 1
            
    return tp, fp, fn

def main():
    csv_file_path = 'C:/Users/Kai/Documents/Time_Series_Anomaly_Detection/Time-Series-Anomaly-Detection-Seminar/TSB-AD/Datasets/TSB-AD-U-Self/012_RAMmedium_12_Hardware_tr_1200_1st_1237.csv'
    data_values = []
    true_labels = []
    
    try:
        with open(csv_file_path, 'r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) == 2: # Ensure row has two columns
                    try:
                        data_values.append(float(row[0]))
                        true_labels.append(int(row[1]))
                    except ValueError:
                        print(f"Skipping invalid row: {row}")
                        # Add placeholders or decide how to handle missing/malformed data
                        # For now, we'll assume data integrity for simplicity in calculation length matching
                        # Or, ensure data_values and true_labels remain synchronized
                        pass 
                else:
                    print(f"Skipping row with unexpected number of columns: {row}")

    except FileNotFoundError:
        print(f"Error: The file {csv_file_path} was not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        return

    if not data_values:
        print("No data was loaded from the CSV file.")
        return

    # Identify anomalies based on data values
    predicted_anomalies = identify_anomalies(data_values)
    
    # Ensure lengths match before calculating metrics
    if len(true_labels) != len(predicted_anomalies):
        print("Mismatch in length between true labels and predicted anomalies. Cannot calculate metrics accurately.")
        # This might happen if some rows were skipped during parsing.
        # A robust solution would align them or handle it based on requirements.
        # For now, we'll proceed if lengths are okay, otherwise print error.
        return

    # Calculate TP, FP, FN
    tp, fp, fn = calculate_metrics(true_labels, predicted_anomalies)
    
    # Calculate Precision and Sensitivity (Recall)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Sensitivity (Recall): {sensitivity:.4f}")

if __name__ == "__main__":
    main()
