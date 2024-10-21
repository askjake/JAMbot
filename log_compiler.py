import os
import re
import json
from collections import defaultdict

# Define the directory containing the log files
log_directory = "C:\\Users\\jacob.montgomery\\Documents\\JAMbot\\logs"
output_json = "aggregated_logs.json"  # The output JSON file to store aggregated data

# Define a regex pattern to extract time-stamp and the rest of the line
pattern = re.compile(r"\[.*\]<(.+?)><.*?>\s*(.*)")

def process_log_file(file_path, line_data):
    """
    Process a single log file to find unique lines and their occurrences with timestamps.
    Updates the provided line_data dictionary with findings.
    """
    # Open and read the file line by line with utf-8 encoding and error handling
    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                timestamp = match.group(1).strip()  # Extract the timestamp
                log_line = match.group(2).strip()   # Extract the actual log message

                # Update the data structure with the extracted line and timestamp
                line_data[log_line]["timestamps"].append(timestamp)
                line_data[log_line]["count"] += 1


def save_to_json(data, output_path):
    """
    Save the processed data to a JSON file.
    """
    with open(output_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def main():
    # Initialize a data structure to store aggregated results across all files
    aggregated_data = defaultdict(lambda: {"timestamps": [], "count": 0})

    # Iterate through each file in the specified directory
    for filename in os.listdir(log_directory):
        file_path = os.path.join(log_directory, filename)

        # Ensure we are processing only files, not directories
        if os.path.isfile(file_path):
            print(f"Processing {filename}...")
            process_log_file(file_path, aggregated_data)

    # Save the aggregated data to a single JSON file
    output_path = os.path.join(log_directory, output_json)
    save_to_json(aggregated_data, output_path)
    print(f"Aggregated data saved to {output_json}")

if __name__ == "__main__":
    main()
