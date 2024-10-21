import json
import os

def extract_relevant_entries(input_file, output_dir, max_summary_size=1024 * 1024):
    """
    Parse the JSON file and extract relevant entries mentioning crashes or failures during the bridge build process.
    If the summary is too large, create more detailed splits and index them for further analysis.

    :param input_file: Path to the input JSON file.
    :param output_dir: Directory where summary and detailed files will be saved.
    :param max_summary_size: Maximum size of the summary file in bytes.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(input_file, 'r') as f:
        data = json.load(f)

    relevant_entries = []
    detailed_files = []
    index = 0

    # Define keywords for crashes or failures during bridge build.
    keywords = ["NCM crash", "bridge build", "failure", "error", "exception", "critical"]

    # Extract relevant entries from the JSON data
    for entry in data:
        # Check for any of the keywords in the entry values (assuming a dictionary structure)
        if any(keyword.lower() in json.dumps(entry).lower() for keyword in keywords):
            relevant_entries.append(entry)

    # Save a summary of the relevant entries
    summary_file = os.path.join(output_dir, 'summary_relevant_entries.json')
    with open(summary_file, 'w') as f:
        json.dump(relevant_entries, f, indent=4)

    # Check if the summary file exceeds the max size allowed
    if os.path.getsize(summary_file) > max_summary_size:
        print(f"Summary file is too large ({os.path.getsize(summary_file)} bytes). Splitting into smaller parts...")
        # Split into smaller files
        chunk_size = max_summary_size // 2  # Define a smaller size for detailed files
        current_chunk = []
        current_size = 0

        for entry in relevant_entries:
            current_chunk.append(entry)
            current_size += len(json.dumps(entry).encode('utf-8'))

            # If the chunk reaches the size limit, save it and reset
            if current_size >= chunk_size:
                chunk_file = os.path.join(output_dir, f'detailed_part_{index}.json')
                with open(chunk_file, 'w') as chunk_f:
                    json.dump(current_chunk, chunk_f, indent=4)
                detailed_files.append(chunk_file)
                index += 1
                current_chunk = []
                current_size = 0

        # Save any remaining entries
        if current_chunk:
            chunk_file = os.path.join(output_dir, f'detailed_part_{index}.json')
            with open(chunk_file, 'w') as chunk_f:
                json.dump(current_chunk, chunk_f, indent=4)
            detailed_files.append(chunk_file)

    # Create an index file to list all parts for further analysis
    index_file = os.path.join(output_dir, 'index.json')
    with open(index_file, 'w') as f:
        json.dump({"summary_file": summary_file, "detailed_files": detailed_files}, f, indent=4)

    print(f"Summary and detailed files saved in: {output_dir}")
    print(f"Index file created at: {index_file}")
    return summary_file, detailed_files


if __name__ == "__main__":
    input_file = "C:\\Users\\jacob.montgomery\\Documents\\JAMboree\\logs\\networking_issue\\aggregated_logs.json"  # Replace with the path to your JSON file
    output_dir = "parsed_logs"  # Directory where the output files will be saved
    extract_relevant_entries(input_file, output_dir)
