import os
import re
import argparse
import numpy as np
import pandas as pd
import json
import paramiko
import gensim
from gensim import corpora
from gensim.models import LdaModel
import pyLDAvis
import pyLDAvis.gensim_models
import logging
import stat
import gzip

# Enable logging for gensim
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def load_credentials(credentials_file):
    """
    Loads SSH credentials from a JSON file.
    """
    try:
        with open(credentials_file, 'r') as file:
            credentials = json.load(file)
        return credentials
    except Exception as e:
        print(f"Error loading credentials from {credentials_file}: {e}")
        return None


def ssh_connect(host, username, password):
    """
    Establishes an SSH connection.
    """
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname=host, username=username, password=password)
        return ssh
    except Exception as e:
        print(f"SSH connection failed: {e}")
        return None


def sftp_list_files(sftp, remote_path, file_filter=None):
    """
    Lists files in a remote directory via SFTP, optionally filtering by a substring.
    """
    all_files = []

    def recursive_list(path):
        try:
            for entry in sftp.listdir_attr(path):
                full_path = os.path.join(path, entry.filename)
                if stat.S_ISDIR(entry.st_mode):
                    recursive_list(full_path)
                elif file_filter in entry.filename:
                    all_files.append(full_path)
        except Exception as e:
            print(f"Error accessing {path}: {e}")

    recursive_list(remote_path)
    return all_files


def read_remote_file(sftp, remote_file_path):
    """
    Reads the content of a remote file via SFTP, decompressing if it is a .gz file.
    """
    try:
        with sftp.open(remote_file_path, 'rb') as remote_file:  # Open in binary mode for gz files
            if remote_file_path.endswith('.gz'):
                with gzip.GzipFile(fileobj=remote_file) as gz_file:
                    content = gz_file.read()
            else:
                content = remote_file.read()
        return content.decode('utf-8', errors='ignore')  # Ensure content is a string
    except Exception as e:
        print(f"Failed to read remote file {remote_file_path}: {e}")
        return None


def parse_log_line(line, log_type):
    """
    Parses a single log line and extracts the message content based on log type.
    """
    if log_type == 'nal':
        pattern = r"\[(.*?)\]\s*(.*)"
    elif log_type == 'netra':
        pattern = r"\[(.*?)\]<.*?><.*?>\s*(.*)"
    elif log_type == 'stbCtrl':
        pattern = r"\[(.*?)\]<.*?><.*?>\s*(.*)"
    else:
        # Default pattern
        pattern = r"\[(.*?)\]<.*?><.*?>\s*(.*)"

    match = re.match(pattern, line)
    if match:
        component = match.group(1)
        message = match.group(2)
        return component, message.strip()
    else:
        return None, None


def read_logs_in_chunks(sftp, log_files, log_type, chunk_size=100000):
    """
    Generator that yields chunks of log messages read from remote files via SFTP.
    """
    log_messages = []
    total_messages = 0
    for log_file in log_files:
        print(f"Processing remote file: {log_file}")
        content = read_remote_file(sftp, log_file)
        if content:
            lines = content.splitlines()
            for line in lines:
                _, message = parse_log_line(line, log_type)
                if message:
                    log_messages.append(message)
                    total_messages += 1
                    if len(log_messages) >= chunk_size:
                        yield log_messages
                        log_messages = []
            print(f"Extracted {total_messages} messages so far.")
        else:
            print(f"No content read from {log_file}")
    if log_messages:
        yield log_messages
    print(f"Total log messages processed: {total_messages}")


def preprocess_messages(messages):
    """
    Preprocesses log messages for topic modeling.
    """
    preprocessed = []
    for msg in messages:
        # Remove IP addresses and numbers to generalize patterns
        msg = re.sub(r'\b\d{1,3}(?:\.\d{1,3}){3}\b', 'IP_ADDRESS', msg)
        msg = re.sub(r'\b\d+\b', 'NUMBER', msg)
        # Tokenize and clean up the message
        tokens = gensim.utils.simple_preprocess(msg, deacc=True)
        preprocessed.append(tokens)
    return preprocessed


def save_visualization(lda_model, corpus, dictionary, output_dir):
    """
    Saves the LDA visualization to the output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not corpus:
        print("Error: Corpus is empty. Cannot create visualization.")
        return

    print("Preparing visualization...")
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis, os.path.join(output_dir, 'lda_visualization.html'))
    print(f"Visualization saved to {os.path.join(output_dir, 'lda_visualization.html')}")


def main():
    parser = argparse.ArgumentParser(description="Log Analysis with Incremental LDA over SSH")
    parser.add_argument('-d', '--directory', type=str, required=True, help='Remote directory containing log files')
    parser.add_argument('-l', '--log_type', type=str, required=True,
                        help='Log type to process (e.g., nal, netra, stbCtrl)')
    parser.add_argument('-c', '--chunk_size', type=int, default=100000, help='Number of log messages per chunk')
    parser.add_argument('-t', '--num_topics', type=int, default=10, help='Number of topics for LDA model')
    parser.add_argument('-p', '--passes', type=int, default=1, help='Number of passes through the corpus during training')
    args = parser.parse_args()

    # Set file_filter and output_dir based on log_type
    args.file_filter = args.log_type
    args.output_dir = os.path.join('logs', args.log_type)

    credentials = load_credentials("/home/montjac/JAMbot/credentials.txt")
    if not credentials:
        return

    ssh_host = credentials.get("linux_pc")
    ssh_username = credentials.get("username")
    ssh_password = credentials.get("password")

    ssh = ssh_connect(ssh_host, ssh_username, ssh_password)
    if not ssh:
        return

    sftp = ssh.open_sftp()
    remote_path = args.directory
    print(f"Processing remote directory: {remote_path}")
    log_files = sftp_list_files(sftp, remote_path, file_filter=args.file_filter)
    print(f"Found {len(log_files)} log files.")

    if len(log_files) == 0:
        print("No log files found. Exiting.")
        sftp.close()
        ssh.close()
        return

    # Build the dictionary
    dictionary = corpora.Dictionary()
    total_messages = 0
    chunk_generator = read_logs_in_chunks(sftp, log_files, args.log_type, chunk_size=args.chunk_size)

    print("Building the dictionary...")
    for messages in chunk_generator:
        preprocessed_messages = preprocess_messages(messages)
        dictionary.add_documents(preprocessed_messages)
        total_messages += len(messages)
        del messages, preprocessed_messages

    print(f"Total messages collected: {total_messages}")
    print(f"Dictionary size: {len(dictionary)}")

    if len(dictionary) == 0:
        print("Error: Dictionary is empty. Exiting.")
        sftp.close()
        ssh.close()
        return

    # Freeze the dictionary to prevent further updates
    dictionary.compactify()

    # Train the LDA model
    lda_model = None
    total_corpus = []
    chunk_generator = read_logs_in_chunks(sftp, log_files, args.log_type, chunk_size=args.chunk_size)
    chunk_count = 0

    for messages in chunk_generator:
        chunk_count += 1
        print(f"Processing chunk {chunk_count} with {len(messages)} messages...")

        preprocessed_messages = preprocess_messages(messages)
        corpus = [dictionary.doc2bow(text) for text in preprocessed_messages]

        if not corpus:
            print(f"Skipping chunk {chunk_count} due to empty corpus.")
            continue

        if lda_model is None:
            print("Initializing LDA model...")
            lda_model = LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=args.num_topics,
                passes=args.passes,
                update_every=0,
                chunksize=args.chunk_size,
                alpha='auto',
                per_word_topics=True
            )
        else:
            print("Updating LDA model...")
            lda_model.update(corpus)

        total_corpus.extend(corpus)
        del messages, preprocessed_messages, corpus

    if lda_model is None:
        print("Error: No valid data processed. Exiting.")
        sftp.close()
        ssh.close()
        return

    save_visualization(lda_model, total_corpus, dictionary, args.output_dir)

    topics = lda_model.print_topics(num_words=10)
    topics_df = pd.DataFrame(topics, columns=['TopicID', 'TopicTerms'])
    topics_df.to_csv(os.path.join(args.output_dir, 'topics_overview.csv'), index=False)
    print(f"Topics overview saved to {os.path.join(args.output_dir, 'topics_overview.csv')}")

    lda_model.save(os.path.join(args.output_dir, 'lda_model.gensim'))
    dictionary.save(os.path.join(args.output_dir, 'dictionary.gensim'))
    print(f"LDA model and dictionary saved to {args.output_dir}")

    sftp.close()
    ssh.close()


if __name__ == "__main__":
    main()
