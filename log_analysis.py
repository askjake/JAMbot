import os
import re
import glob
import argparse
import pandas as pd
from tqdm import tqdm
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer

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

def read_logs(log_dir):
    """
    Reads all log files from the specified directory and returns a list of log messages.
    """
    log_messages = []
    log_files = glob.glob(os.path.join(log_dir, "*.*"))
    print(f"Found {len(log_files)} log files.")

    for log_file in log_files:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                _, message = parse_log_line(line)
                if message:
                    log_messages.append(message)
    print(f"Extracted {len(log_messages)} log messages.")
    return log_messages

def preprocess_messages(messages):
    """
    Preprocesses log messages for topic modeling.
    """
    # Basic preprocessing: you can expand this as needed
    preprocessed = []
    for msg in messages:
        # Remove IP addresses and numbers to generalize patterns
        msg = re.sub(r'\b\d{1,3}(?:\.\d{1,3}){3}\b', 'IP_ADDRESS', msg)
        msg = re.sub(r'\b\d+\b', 'NUMBER', msg)
        preprocessed.append(msg)
    return preprocessed

def generate_embeddings(messages, model_name='all-MiniLM-L6-v2'):
    """
    Generates embeddings for the messages using a sentence transformer model.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(messages, show_progress_bar=True)
    return embeddings

def perform_topic_modeling(messages, embeddings):
    """
    Performs topic modeling using BERTopic and returns the model and topics.
    """
    # You can customize the vectorizer and other parameters
    vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words='english')
    topic_model = BERTopic(vectorizer_model=vectorizer_model)
    topics, probs = topic_model.fit_transform(messages, embeddings)
    return topic_model, topics, probs

def save_visualizations(topic_model, output_dir):
    """
    Saves interactive visualizations to the output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    topics_fig = topic_model.visualize_topics()
    topics_fig.write_html(os.path.join(output_dir, 'topics.html'))

    hierarchy_fig = topic_model.visualize_hierarchy()
    hierarchy_fig.write_html(os.path.join(output_dir, 'hierarchy.html'))

    heatmap_fig = topic_model.visualize_heatmap()
    heatmap_fig.write_html(os.path.join(output_dir, 'heatmap.html'))

    barchart_fig = topic_model.visualize_barchart()
    barchart_fig.write_html(os.path.join(output_dir, 'barchart.html'))

    print(f"Visualizations saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Log Analysis with BERTopic")
    parser.add_argument('--log_dir', type=str, required=True, help='Directory containing log files')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save visualizations')
    parser.add_argument('--model_name', type=str, default='all-MiniLM-L6-v2', help='SentenceTransformer model name')
    args = parser.parse_args()

    print("Reading log files...")
    messages = read_logs(args.log_dir)

    print("Preprocessing messages...")
    preprocessed_messages = preprocess_messages(messages)

    print("Generating embeddings...")
    embeddings = generate_embeddings(preprocessed_messages, model_name=args.model_name)

    print("Performing topic modeling...")
    topic_model, topics, probs = perform_topic_modeling(preprocessed_messages, embeddings)

    print("Saving visualizations...")
    save_visualizations(topic_model, args.output_dir)

    # Optional: Save topics to a CSV file
    topics_overview = topic_model.get_topic_info()
    topics_overview.to_csv(os.path.join(args.output_dir, 'topics_overview.csv'), index=False)
    print(f"Topics overview saved to {args.output_dir}/topics_overview.csv")

if __name__ == "__main__":
    main()
