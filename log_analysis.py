import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

def parse_log_line(line):
    """
    Parses a single log line and extracts the message content.
    """
    # Regular expression pattern to extract the log message
    pattern = r"\[(.*?)\]<.*?><.*?>\s*(.*)"
    match = re.match(pattern, line)
    if match:
        # You can extract more fields if needed
        component = match.group(1)
        message = match.group(2)
        return component, message.strip()
    else:
        return None, None

def read_logs_in_chunks(log_files, chunk_size=100000):
    """
    Generator that yields chunks of log messages.
    """
    log_messages = []
    total_messages = 0
    for log_file in log_files:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                _, message = parse_log_line(line)
                if message:
                    log_messages.append(message)
                    total_messages += 1
                    if len(log_messages) >= chunk_size:
                        yield log_messages
                        log_messages = []
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
        preprocessed.append(msg)
    return preprocessed

def generate_embeddings(messages, model):
    """
    Generates embeddings for the messages using a sentence transformer model.
    """
    embeddings = model.encode(messages, show_progress_bar=True)
    embeddings = normalize(embeddings)
    return embeddings

def save_visualizations(topic_model, output_dir):
    """
    Saves interactive visualizations to the output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Visualize topics
    topics_fig = topic_model.visualize_topics()
    topics_fig.write_html(os.path.join(output_dir, 'topics.html'))

    # Visualize hierarchy with error handling
    try:
        hierarchy_fig = topic_model.visualize_hierarchy(distance_function='euclidean')
        hierarchy_fig.write_html(os.path.join(output_dir, 'hierarchy.html'))
    except ValueError as e:
        print(f"Could not generate hierarchy visualization: {e}")
        print("Attempting to visualize hierarchy with a sample of the data...")
        # Use a smaller sample
        sample_size = 10000
        indices = np.random.choice(len(topic_model.documents), size=sample_size, replace=False)
        sample_embeddings = topic_model.embedding_model.transform([topic_model.documents[i] for i in indices])
        sample_embeddings = normalize(sample_embeddings)
        sample_topic_model = BERTopic()
        sample_topics, _ = sample_topic_model.fit_transform([topic_model.documents[i] for i in indices], sample_embeddings)
        # Visualize hierarchy on the sample
        hierarchy_fig = sample_topic_model.visualize_hierarchy(distance_function='euclidean')
        hierarchy_fig.write_html(os.path.join(output_dir, 'hierarchy.html'))

    # Visualize heatmap
    heatmap_fig = topic_model.visualize_heatmap()
    heatmap_fig.write_html(os.path.join(output_dir, 'heatmap.html'))

    # Visualize barchart
    barchart_fig = topic_model.visualize_barchart()
    barchart_fig.write_html(os.path.join(output_dir, 'barchart.html'))

    print(f"Visualizations saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Log Analysis with BERTopic")
    parser.add_argument('--log_dir', type=str, required=True, help='Directory containing log files')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save visualizations')
    parser.add_argument('--model_name', type=str, default='all-MiniLM-L6-v2', help='SentenceTransformer model name')
    parser.add_argument('--chunk_size', type=int, default=100000, help='Number of log messages per chunk')
    args = parser.parse_args()

    # Initialize variables
    all_embeddings = []
    all_messages = []

    print("Reading log files...")
    log_files = glob.glob(os.path.join(args.log_dir, "*.*"))
    print(f"Found {len(log_files)} log files.")

    model = SentenceTransformer(args.model_name)

    # Process logs in chunks
    chunk_generator = read_logs_in_chunks(log_files, chunk_size=args.chunk_size)
    chunk_count = 0
    for messages in chunk_generator:
        chunk_count += 1
        print(f"Processing chunk {chunk_count} with {len(messages)} messages...")

        # Preprocess messages
        preprocessed_messages = preprocess_messages(messages)

        # Generate embeddings
        embeddings = generate_embeddings(preprocessed_messages, model)

        # Append to all embeddings and messages
        all_embeddings.append(embeddings)
        all_messages.extend(preprocessed_messages)

        # Optional: Free memory if necessary
        del messages, preprocessed_messages, embeddings

    # Concatenate all embeddings
    all_embeddings = np.vstack(all_embeddings)

    print(f"Total messages collected: {len(all_messages)}")
    print(f"Total embeddings shape: {all_embeddings.shape}")

    print("Performing topic modeling...")
    # You can customize the vectorizer and other parameters
    vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words='english')
    topic_model = BERTopic(vectorizer_model=vectorizer_model)
    topics, probs = topic_model.fit_transform(all_messages, all_embeddings)

    print("Saving visualizations...")
    save_visualizations(topic_model, args.output_dir)

    # Optional: Save topics to a CSV file
    topics_overview = topic_model.get_topic_info()
    topics_overview.to_csv(os.path.join(args.output_dir, 'topics_overview.csv'), index=False)
    print(f"Topics overview saved to {args.output_dir}/topics_overview.csv")

if __name__ == "__main__":
    main()
