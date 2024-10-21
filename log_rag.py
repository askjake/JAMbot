import os
import json
import logging
import argparse
from collections import defaultdict
from drain3 import TemplateMiner
import math
import requests
import gzip
import re
import tarfile
import numpy as np
import psycopg2  # Assuming you're using PostgreSQL
import codecs
import time
import paramiko

# Set up logging
logging.basicConfig(filename='debugJam.log', level=logging.DEBUG)

paramiko_logger = logging.getLogger("paramiko")
paramiko_logger.setLevel(logging.ERROR)  # Only show warnings and errors for Paramiko
# Suppress DEBUG logs from urllib3
urllib3_logger = logging.getLogger("urllib3")
urllib3_logger.setLevel(logging.WARNING)  # You can also use ERROR if you want to suppress warnings too


# Set directories
script_dir = 'scripts'
log_directory = 'logs'
response_directory = 'ai_responses'
interim_directory = os.path.join(response_directory, 'brainbase')
brainbase = os.path.join(response_directory, 'brainbase') 
brainbed = os.path.join(response_directory, 'brainbed')  

credentials_file = 'scripts/credentials.txt'
# Initialize Drain3 template miner for log parsing
template_miner = TemplateMiner()

# URLs for embedding and API
aqua_embed_url = 'http://localhost:5000/embed'
aqua_api_url = 'http://localhost:5000/ollama'

# Function to load credentials from a file
def load_credentials():
    try:
        with open(credentials_file, 'r') as file:
            credentials = json.load(file)
            logging.info("Credentials loaded successfully.")
            return credentials
    except Exception as e:
        logging.error(f"Error loading credentials from {credentials_file}: {str(e)}")
        return None


# Modify the log path by replacing invalid characters (such as colons or slashes) to ensure compatibility with the operating system's file system.
def sanitize_path(log_path):
    # Replace colons and other invalid characters for Windows file system
    log_path = log_path.replace(":", "-")
    log_path = log_path.replace("/", os.sep)  # Ensure correct slashes for the current OS
    return log_path

# Function to fetch logs from a Linux PC using SSH if they are not already present locally.
def fetch_logs_from_linux_pc(log_path):
    credentials = load_credentials()
    if credentials is None:
        logging.error("No credentials loaded, cannot fetch logs.")
        return None

    username = credentials['username']
    password = credentials['password']
    linux_pc = credentials['linux_pc']

    try:
        # Strip "/ccshare/logs/smplogs/" and retain the rest of the directory structure
        relative_log_path = os.path.relpath(log_path, "/ccshare/logs/smplogs")
        destination_path = os.path.join(interim_directory, sanitize_path(relative_log_path))

        # Check if the file already exists locally
        if os.path.exists(destination_path):
            logging.info(f"Log {destination_path} already exists locally, skipping download.")
            return destination_path

        # If not, proceed with fetching the log
        logging.info(f"Connecting to {linux_pc} via SSH to fetch logs.")
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(linux_pc, username=username, password=password)

        sftp = ssh.open_sftp()

        # Ensure the local directory structure exists
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)

        logging.info(f"Fetching {log_path} from Linux PC to {destination_path}.")
        sftp.get(log_path, destination_path)

        sftp.close()
        ssh.close()

        logging.info(f"Successfully fetched log: {log_path}")
        return destination_path  # Return the path where the log is saved locally
    except Exception as e:
        logging.error(f"Failed to fetch log from {linux_pc}: {str(e)}")
        return None
        
# Function to read log files, including .log, .gz (gzip compressed), and tar.gz files, while handling various formats gracefully.
def read_log_file(file_path):
    try:
        if file_path.endswith(".log"):
            logging.debug(f"Reading log file: {file_path}")
            with open(file_path, 'r', encoding='utf-8', errors='replace') as log_file:
                return log_file.read()
        elif file_path.endswith(".gz"):  # For gzipped log files
            logging.debug(f"Reading gzip file: {file_path}")
            if tarfile.is_tarfile(file_path):  # Handle .tar.gz files
                with tarfile.open(file_path, 'r:gz') as tar:
                    for tar_info in tar:
                        if tar_info.isfile() and tar_info.name.endswith(".log"):  # Extract and process .log files
                            logging.debug(f"Extracting file from tar.gz: {tar_info.name}")
                            log_file = tar.extractfile(tar_info)
                            return log_file.read().decode('utf-8', errors='replace')
            else:  # Handle standalone .gz files
                logging.debug(f"Opening standalone .gz file: {file_path}")
                with gzip.open(file_path, 'rt', encoding='utf-8', errors='replace') as log_file:  # 'rt' mode with 'utf-8' encoding
                    return log_file.read()
        else:
            # For all other file types, read them as plain text files
            logging.debug(f"Reading unrecognized file type as text: {file_path}")
            with open(file_path, 'r', encoding='utf-8', errors='replace') as unknown_file:
                return unknown_file.read()
    except Exception as e:
        logging.error(f"Failed to read file {file_path}: {str(e)}")
        return None

# Function to extract .gz file paths from residual log lines, identifying the relevant log files from the provided line content.
def extract_log_paths_from_residual(line):
    # Implement a method to parse the residual log lines and extract log file paths
    # Example:
    paths = []
    if "gz" in line:
        parts = line.split()
        for part in parts:
            if part.endswith('.gz'):
                paths.append(part)
    return paths
    
    # Function to save an entire log file to an interim directory, preserving its original structure, and delete the original file afterward.
def save_entire_log_to_interim(file_path, log_name):
    # Get the relative path of the file from the log_directory
    relative_path = os.path.relpath(file_path, log_directory)

    # Create the interim path using the relative path
    interim_file_path = os.path.join(interim_directory, os.path.dirname(relative_path), log_name)

    # Ensure the interim directory structure exists
    os.makedirs(os.path.dirname(interim_file_path), exist_ok=True)

    try:
        # Use UTF-8 encoding explicitly when reading and writing the file
        with open(file_path, 'r', encoding='utf-8', errors='replace') as original_file:
            content = original_file.read()
        with open(interim_file_path, 'w', encoding='utf-8', errors='replace') as interim_file:
            interim_file.write(content)
        
        logging.debug(f"Saved log to: {interim_file_path}")
        print(f"Saved log to: {interim_file_path}")
        
        # Delete the original file after saving to interim
        os.remove(file_path)
        logging.debug(f"Deleted original file: {file_path}")
        print(f"Deleted original file: {file_path}")

    except Exception as e:
        logging.error(f"Failed to save file to interim: {e}")
        print(f"Failed to save file to interim: {e}")

        
# Function to split logs by category, extracting categories from log lines and grouping the lines into relevant categories.
def split_logs_by_category(content, file_path, log_name):
    logging.debug(f"Splitting log content by category for log: {log_name}")
    
    category_logs = defaultdict(list)
    lines = content.splitlines()
    
    for line in lines:
        if line.startswith('!['):  # Assuming format: [category]<date time><process> info
            category = line.split(']')[0] + ']'  # Extracting the category
            category_logs[category].append(line)
        else:
            # Stop processing and save the file to interim if poorly formatted
            logging.warning(f"Poorly formatted line found in {log_name}. Stopping processing.")
            save_entire_log_to_interim(file_path, log_name)
            return None  # Early exit
    
    # Save logs for each category
    save_category_logs(category_logs, log_name)

# Function to save the categorized logs into interim files, maintaining the original directory structure and saving logs for each category separately.
def save_category_logs(category_logs, log_name):
    # Get the relative path of the file from the log_directory
    relative_path = os.path.relpath(log_name, log_directory)
    
    # Create the interim directory structure using the relative path
    interim_file_dir = os.path.join(interim_directory, os.path.dirname(relative_path))  # Retain original directory structure
    os.makedirs(interim_file_dir, exist_ok=True)
    
    for category, logs in category_logs.items():
        interim_file_path = os.path.join(interim_file_dir, f"{category}_interim.log")
        
        try:
            with open(interim_file_path, 'w') as file:
                file.write('\n'.join(logs))
            logging.debug(f"Saved interim log file for category {category}: {interim_file_path}")
            print(f"Saved interim log file for category {category}: {interim_file_path}")
        except Exception as e:
            logging.error(f"Failed to save category log {category}: {e}")

# Function to parse and groom logs using the Drain algorithm, converting raw log lines into structured templates for further analysis.
def groom_logs_with_drain(content):
    logging.debug(f"Grooming log content")
    parsed_lines = []
    
    # Split content by lines (assumes each log message is on a separate line)
    lines = content.splitlines()
    
    for line in lines:
        # Parse log line using Drain algorithm
        result = template_miner.add_log_message(line)
        template = result['template_mined']
        parsed_lines.append(template)

    logging.debug(f"Parsed log content into {len(parsed_lines)} templates.")
    print(f"Parsed log content into {len(parsed_lines)} lines.")
    return "\n".join(parsed_lines)

# Function to split the log content into smaller chunks, based on a specified word count, for easier processing and sending to the API.
def split_log_into_chunks(log_content, chunk_size):
    words = log_content.split()
    
    # Avoid splitting if the log is small enough to handle in one piece.
    if len(words) <= chunk_size:
        logging.debug("Log content is small enough to process as a single chunk.")
        return [log_content]
    
    total_chunks = math.ceil(len(words) / chunk_size)
    logging.debug(f"Splitting log content into {total_chunks} chunks")
    print(f"Splitting log content into {total_chunks} chunks")
    chunks = [" ".join(words[i * chunk_size:(i + 1) * chunk_size]) for i in range(total_chunks)]
    return chunks


# Function to send log chunks to the Aqua API for analysis, packaging the prompt and sending requests to the specified model and endpoint.
def send_to_aqua(prompt, model="Aqua", template=None, chunk_size=4089, use_embed=False, max_tokens=512):
    num_ctx = max(4089, chunk_size * 2)  # Adjust num_ctx based on chunk size
    headers = {'Content-Type': 'application/json'}
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "num_ctx": str(num_ctx),
        "history": True,
        "template": template,
        "max_tokens": max_tokens
    }

    target_url = aqua_embed_url if use_embed else aqua_api_url

    try:
        logging.debug(f"Sending request to {target_url} with prompt length {len(prompt)} characters, using model: {model}, num_ctx: {num_ctx}, max_tokens: {max_tokens}")
        print(f"Sending request to {target_url} API with prompt length {len(prompt)} characters")
        response = requests.post(target_url, headers=headers, json=data)
        if response.status_code == 200:
            response_data = response.json()
            detailed_response = response_data.get('response', 'No response received')
            #logging.debug(f"Received response from {model}: {detailed_response}")
            #print(f"Received response from {model}: {detailed_response}")
            return detailed_response
        else:
            error_message = f"Error: Received status code {response.status_code} with message: {response.text}"
            logging.error(error_message)
            print(error_message)
            return None
    except requests.RequestException as e:
        error_message = f"Failed to connect to {model}: {str(e)}"
        logging.error(error_message)
        print(error_message)
        return None


# Function to process logs by splitting them into chunks and sending each chunk to Aqua, handling batch processing and delays between API calls.
def process_and_send_logs(parsed_content, file_path, filename, chunk_size, batch_size=10, delay=0, model="Aqua", use_embed=False):
    logging.debug(f"Processing log: {filename}")
    print(f"Processing log: {filename}")
    
    chunks = split_log_into_chunks(parsed_content, chunk_size)
    responses = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        
        for j, chunk in enumerate(batch):
            prompt = f"Please analyze the following log file chunk {i+j+1}/{len(chunks)}:\n\n{chunk}"
            template = """
                <|im_start|>system
                Aqua is a highly knowledgeable troubleshooting assistant, with a focus on technical precision and clarity. Aqua understands complex systems, including media playback, network communications, session management, and digital broadcasting systems, and breaks down issues methodically. Aqua’s insights are practical, clear, and detail-rich, providing actionable advice tailored to resolving system errors efficiently. Aqua never assumes prior knowledge and adjusts responses based on user expertise, offering both high-level summaries and deep dives into technical details when needed. Aqua is factual, detail-oriented, and patient, following up with clarifying questions if more context is required.
                <|im_end|>
                <|im_start|>user
                {{ .Prompt }}
                <|im_end|>
                <|im_start|>assistant
            """
            
            logging.debug(f"Sending chunk {i+j+1}/{len(chunks)} of {filename} to Chat API")
            print(f"Sending chunk {i+j+1}/{len(chunks)} of {filename} to Aqua API")
            
            try:
                response = send_to_aqua(prompt, model=model, template=template, chunk_size=chunk_size, use_embed=use_embed)
                
                if response:
                    logging.debug(f"Response received for chunk {i+j+1}/{len(chunks)} of {filename}")
                    print(f"Response received for chunk {i+j+1}/{len(chunks)} of {file_path}\n")
                    responses.append({"chunk": i+j+1, "response": response})
                    save_responses(file_path, filename, responses, model, use_embed)
                else:
                    raise Exception(f"Failed to get a response for chunk {i+j+1}/{len(chunks)} of {filename}")
            
            except Exception as e:
                logging.error(f"Error processing chunk {i+j+1}: {e}")
                print(f"Error processing chunk {i+j+1}: {e}")
                continue  # Continue processing the next chunks even if one fails

        # Save responses regularly in case of failure
        save_responses(file_path, filename, responses, model, use_embed)

        time.sleep(delay)


# Function to save the responses from Aqua API into a structured JSON file, preserving the original file structure and response data.
def save_responses(file_path, filename, responses, model, use_embed):
    # Remove 'interim' from the directory path
    relative_path = os.path.relpath(file_path, interim_directory).replace("interim", "")
    response_dir = os.path.join(response_directory, os.path.dirname(relative_path))
    os.makedirs(response_dir, exist_ok=True)

    # Choose the appropriate suffix based on the API used
    suffix = "embedded" if use_embed else "ragged"

    # Generate response filename without 'interim'
    response_filename = f"{filename}_{model}_{suffix}.json".replace("_interim", "")
    response_filepath = os.path.join(response_dir, response_filename)

    # Save the responses
    with open(response_filepath, 'w') as response_file:
        json.dump(responses, response_file, indent=4)

    logging.debug(f"Responses saved to {response_filepath}")

def extract_metadata(log_content, model='Aqua', part=None, summary_max_tokens=1024):
    try:
        # Extract timestamp, error type, and file size information
        timestamp_match = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', log_content)
        error_type_match = re.search(r'Error|Warning|Critical', log_content, re.IGNORECASE)
        file_sizes = re.findall(r'has a file size of (\d+) KB', log_content)
        total_file_size_kb = sum([float(size) for size in file_sizes])

        metadata = {
            'timestamp': timestamp_match.group() if timestamp_match else 'unknown',
            'error_type': error_type_match.group() if error_type_match else 'unknown',
            'total_file_size_kb': total_file_size_kb,
            'log_length': len(log_content),
            'part': part,
        }

        # Prepare the prompt for Aqua and truncate the log content if needed
        max_chars = 100000  # Adjust based on API limitations
        truncated_log_content = log_content[:max_chars]
        prompt = f"Please provide a concise summary of the following log content (part {part}):\n\n{truncated_log_content}"

        # Send to Aqua API to get the summary
        summary = send_to_aqua(prompt, model=model, max_tokens=summary_max_tokens)
        metadata['summary'] = summary.strip()[:1024] if summary else 'No summary available'

        # Log the metadata for debugging
        logging.debug(f"metadata: {metadata}")
        print(f"Metadata extracted: {metadata}")

        # Save the metadata to the file (optional, for persistence)
        metadata_file_path = os.path.join(response_directory, 'metadata')
        os.makedirs(response_directory, exist_ok=True)
        with open(metadata_file_path, 'a') as f:
            json.dump(metadata, f)
            f.write('\n')
        logging.debug(f"Appended metadata to {metadata_file_path}")

        # Return metadata including the summary directly for further use
        return metadata

    except Exception as e:
        logging.error(f"Failed to extract metadata: {e}")
        return None

    
def embed_log(log_content):
    # Send log to embedding API
    payload = {
        "model": "Aqua",  # or your preferred model
        "prompt": log_content,
        "truncate": True
    }
    response = requests.post('http://localhost:5001/embed', json=payload)
    
    if response.status_code == 200:
        embeddings = response.json().get('response')
        return embeddings
    else:
        raise ValueError("Failed to fetch embeddings")
    
def store_embedding(embedding, metadata):
    vector = np.array(embedding, dtype='float32')
    vector = vector / np.linalg.norm(vector)

    # Use a default timestamp if None
    timestamp = metadata['timestamp'] if metadata['timestamp'] != 'unknown' else '1970-01-01 00:00:00'

    conn = psycopg2.connect("host=10.79.85.40 dbname=chatbotdb user=chatbotuser password=changeme")
    cursor = conn.cursor()

    insert_query = """
    INSERT INTO embeddings (embedding_vector, timestamp, error_type, summary, part)
    VALUES (%s, %s, %s, %s, %s)
    """
    cursor.execute(insert_query, (vector.tolist(), timestamp, metadata.get('error_type', 'N/A'), metadata.get('summary', ''), metadata.get('part', '')))

    conn.commit()
    cursor.close()
    conn.close()

    logging.debug(f"Stored embedding and metadata in vector DB")

def search_similar_logs(new_log_content):
    new_embedding = embed_log(new_log_content)
    
    # Retrieve from FAISS index
    k = 5  # Number of nearest neighbors
    distances, indices = index.search(new_embedding, k)

    # Retrieve metadata from relational DB using indices
    conn = psycopg2.connect("host=10.79.85.40 dbname=chatbotdb user=montjac password=changeme")
    cursor = conn.cursor()
    
    similar_logs = []
    for idx in indices[0]:
        cursor.execute("SELECT * FROM log_embeddings WHERE id = %s", (idx,))
        similar_logs.append(cursor.fetchone())
    
    cursor.close()
    conn.close()

    return similar_logs

def combine_logs(response_directory, combined_directory):
    # Ensure the combined directory exists
    os.makedirs(combined_directory, exist_ok=True)

    # Function to get the base name by stripping the sequence number (e.g., NetConMgr.#_<description>.json -> NetConMgr_<description>.json)
    def get_base_name(filename):
        # Split filename by dots and underscores, and return everything except the sequence number part
        parts = filename.split('_')
        if len(parts) > 1 and len(parts[0]) > 0 and parts[0][-1].isdigit():
            return parts[0][:-1] + '_' + '_'.join(parts[1:])
        return filename

    # Group files by their base name
    file_groups = defaultdict(list)

    # Walk through the files in the response directory
    for root, dirs, files in os.walk(response_directory):
        for file in files:
            if file.endswith(".json"):
                base_name = get_base_name(file)
                file_groups[base_name].append(os.path.join(root, file))

    # Function to combine the files
    def combine_files(file_list, combined_file_path):
        combined_data = []
        
        for file_path in file_list:
            with open(file_path, 'r') as f:
                data = json.load(f)
                combined_data.extend(data)  # Assuming each JSON file contains a list, extend the combined data

        # Save the combined data
        with open(combined_file_path, 'w') as combined_file:
            json.dump(combined_data, combined_file, indent=4)

    # Process each group and combine the files
    for base_name, files in file_groups.items():
        combined_file_path = os.path.join(combined_directory, base_name)  # Combined file path without sequence number
        print(f"Combining {len(files)} files into {combined_file_path}")
        combine_files(files, combined_file_path)

    print(f"All files have been combined and saved in {combined_directory}")
    
def process_file(file_path, chunk_size=100000, model="Alex", use_embed=True):
    """Process individual file"""
    output_messages = []
    logging.debug(f"Processing file: {file_path}")
    content = read_log_file(file_path)

    if content is None or len(content.strip()) == 0:
        message = f"Content is empty or None for {file_path}. Skipping file."
        logging.error(message)
        output_messages.append(message)
        return output_messages

    chunks = split_log_into_chunks(content, chunk_size)
    if not chunks or len(chunks) == 0:
        message = f"Failed to split content into chunks for {file_path}. Skipping file."
        logging.error(message)
        output_messages.append(message)
        return output_messages

    batch_size = 10
    template = 'Alex'
    total_chunks = len(chunks)

    try:
        # Process chunks in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            for j, chunk in enumerate(batch):
                part_number = i + j + 1  # Current chunk number
                part = f"{part_number}/{total_chunks}"
                prompt = f"{part}:\n\n{chunk}"
                message = f"Sending chunk {part} of {file_path} to Aqua API"
                logging.debug(message)
                output_messages.append(message)

                try:
                    embedding = send_to_aqua(prompt, model=model, template=template, chunk_size=chunk_size, use_embed=use_embed)
                    metadata = extract_metadata(
                        chunk, model=model, part=part, summary_max_tokens=200
                    )
                    message = f"Metadata  for chunk {part} of {file_path}"
                    logging.debug(message)
                    logging.debug(metadata)
                    output_messages.append(message)
                    output_messages.append(metadata)

                    if embedding:
                        message = f"Embedding received for chunk {part} of {file_path}\n"
                        logging.debug(message)
                        output_messages.append(message)

                        store_embedding(embedding, metadata)
                        message = f"Successfully stored embedding for {file_path}."
                        logging.debug(message)
                        output_messages.append(message)
                    else:
                        message = f"Failed to get a response for chunk {part} of {file_path}"
                        logging.error(message)
                        output_messages.append(message)
                        raise Exception(message)

                except Exception as e:
                    message = f"Error while processing chunk {part} of {file_path}: {e}"
                    logging.error(message)
                    output_messages.append(message)

    except Exception as e:
        message = f"Error while processing {file_path}: {e}"
        logging.error(message)
        output_messages.append(message)

    return output_messages


# Function to list files from a Linux PC using SSH if the directory is on Linux
def list_files_from_linux_pc(directory_path):
    credentials = load_credentials()
    if credentials is None:
        logging.error("No credentials loaded, cannot list files.")
        return None

    username = credentials['username']
    password = credentials['password']
    linux_pc = credentials['linux_pc']

    try:
        logging.info(f"Connecting to {linux_pc} via SSH to list files in {directory_path}.")
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(linux_pc, username=username, password=password)

        sftp = ssh.open_sftp()

        # List files in the remote directory
        files = sftp.listdir(directory_path)
        file_paths = [os.path.join(directory_path, file) for file in files]
        logging.info(f"Files found in {directory_path}: {file_paths}")

        sftp.close()
        ssh.close()

        return file_paths
    except Exception as e:
        logging.error(f"Failed to list files from {linux_pc}: {str(e)}")
        return None

# Function to fetch logs or files from Linux PC if necessary
def fetch_and_process_remote_directory(remote_directory, chunk_size, model, use_embed):
    file_paths = list_files_from_linux_pc(remote_directory)
    if file_paths:
        for file_path in file_paths:
            # Fetch and process each file
            fetched_path = fetch_logs_from_linux_pc(file_path)
            if fetched_path:
                process_file(fetched_path, chunk_size, model, use_embed)

# Main function to drive the log processing pipeline, handling both local and remote files
def main():
    parser = argparse.ArgumentParser(description="Process log files and send them to Aqua chatbot.")
    parser.add_argument('-m', '--model', type=str, default='Alex', help='The model to use. Default is dolphin-mixtral.')
    parser.add_argument('-c', '--chunk', type=int, default=25000, help='Chunk size in words. Default is 4089.')
    parser.add_argument('-e', '--embed', action='store_true', help="Use the embedding API and store in database.")
    parser.add_argument('-a', '--append', action='store_true', help="Append and combine the resulting logs after processing.")
    parser.add_argument('-f', '--file', type=str, help="Specify a file or directory to process instead of log_directory.")

    args = parser.parse_args()

    model = args.model
    chunk_size = args.chunk
    use_embed = args.embed
    append_logs = args.append
    file_or_directory = args.file if args.file else log_directory

    logging.debug(f"Log processing initiated with model: {model}, chunk size: {chunk_size}, embed flag: {use_embed}, file or directory: {file_or_directory}")

    # Check if the file_or_directory is remote or local
    if file_or_directory.startswith('/ccshare'):
        logging.debug(f"Remote directory detected: {file_or_directory}")
        fetch_and_process_remote_directory(file_or_directory, chunk_size, model, use_embed)
    else:
        # If it's a local file or directory
        if os.path.isfile(file_or_directory):
            process_file(file_or_directory, chunk_size, model, use_embed)
        else:
            for root, dirs, files in os.walk(file_or_directory):
                logging.debug(f"Checking directory: {root}")
                for file in files:
                    file_path = os.path.join(root, file)
                    process_file(file_path, chunk_size, model, use_embed)

    # Handle appending and combining logs
    if append_logs:
        logging.debug(f"Combining logs into {brainbed}")
        combined_logs = "combined_logs.json"  # Define a filename
        combined_file_path = os.path.join(brainbed, combined_logs)  # Save to a file in brainbed
        combine_logs(response_directory, brainbed)  # Ensure it points to a specific file

        combined_content = read_log_file(combined_file_path)  # Ensure you're reading the file correctly
        
        if combined_content:
            metadata = extract_metadata(combined_content, model=model, part="Combined")

if __name__ == "__main__":
    main()