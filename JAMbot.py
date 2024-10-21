# Import necessary modules
from flask import Flask, request, jsonify, render_template, Response
import requests
import numpy as np
import logging
import time
import platform
from web_search import *
import os
import subprocess
import re
import json
from sklearn.metrics.pairwise import cosine_similarity
import psycopg2
from functools import lru_cache
from datetime import datetime  # Add this import
from datetime import timedelta
from threading import Timer
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from grasshopper_access import *

from embed_content import *

# Global variables to control AI behavior
disco_biscuit_active = False

# Set up requests session with retry mechanism
session = requests.Session()
retries = Retry(total=3, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504])
session.mount('http://', HTTPAdapter(max_retries=retries))

# Model host mapping for easy IP selection
MODEL_HOST_MAPPING = {
    "ai1": "10.79.85.40",
    "aqua": "10.79.85.40",
    "gabriel": "10.79.85.40",
    "alex": "10.79.85.40",
    "ai2": "10.79.85.47",
    "gemma": "10.79.85.47",
    "gemma2": "10.79.85.47",
    "avery": "10.79.85.47"    
}

# Set up logging to file
logging.basicConfig(
    filename='logJAM.txt',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # Allow up to 100MB (adjust as needed)


# Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


    
@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    try:
        # Check if an image file is present in the request
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Save the uploaded image temporarily
        file_path = os.path.join('/tmp', file.filename)
        file.save(file_path)

        # Load the image
        image = Image.open(file_path)

        # Use BLIP to generate caption
        inputs = processor(image, return_tensors="pt")
        outputs = model.generate(**inputs)
        caption = processor.decode(outputs[0], skip_special_tokens=True)

        # Remove the temporary file
        os.remove(file_path)

        # Return the generated caption
        return jsonify({"result": caption}), 200

    except Exception as e:
        logging.error(f"Error analyzing image: {str(e)}")
        return jsonify({"error": str(e)}), 500
        
# Cache embeddings for efficiency
@lru_cache(maxsize=1000)
def get_embedding_cache(prompt):
    """
    Retrieve or compute the embedding for a given prompt, using caching.
    """
    logging.info(f"Fetching cached embedding for prompt: {prompt[:50]}...")
    host_ip = MODEL_HOST_MAPPING.get("aqua")  # Default to Aqua for embedding
    try:
        embedding = embed_query({"prompt": prompt, "model": "Aqua"}, host_ip, store_in_db=False)
        return embedding
    except ValueError as e:
        logging.error(f"Error getting cached embedding: {str(e)}")
        return None

# Health check endpoint to verify server status
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "OK"}), 200

# Render chat interface
@app.route('/chat', methods=['GET'])
def chat():
    return render_template('Chatbot.html')

@app.route('/web-search', methods=['POST'])
def web_search():
    data = request.get_json()
    query = data.get("query")
    if not query:
        return jsonify({"error": "Query is required."}), 400

    search_result = do_web_search(query)
    if search_result and not search_result.startswith("An error occurred"):
        return jsonify({"response": search_result}), 200
    else:
        return jsonify({"response": "No results found."}), 200


# Handle the chatbot conversation
@app.route('/ollama', methods=['POST'])
def handle_conversation():
    """
    Handle user conversation with Aqua, Gemma, or other models, and reference embedded logs when needed.
    Ensure that the response is clear, fully answers the user's question, and provides a concrete method of procedure if applicable.
    """
    try:
        # Extract JSON data from the incoming request
        data = request.get_json()

        if not data:
            return jsonify({"error": "Invalid request, JSON data missing."}), 400
        # Extract required parameters with default fallbacks
        user_input = data.get("prompt")
        model = data.get("model", "Alex")
        search_embeddings = data.get("search_embeddings", "no")
        options = data.get("options", {})  # Ensure 'options' is always a dictionary
        
        # Additional option parsing with defaults
        num_predict = options.get("num_predict", 50)
        temperature = options.get("temperature", 1)

        # Validate the presence of user input
        if not user_input:
            return jsonify({"error": "Prompt is required."}), 400

        # Convert input to lowercase for processing
        user_input_lower = user_input.lower()

        # Ensure valid model and resolve its IP address
        model_lower = model.lower()
        host_ip = MODEL_HOST_MAPPING.get(model_lower)
        if not host_ip:
            return jsonify({"error": f"Unknown model specified: {model}"}), 400

        # Embed the user query without storing it
        try:
            query_embedding = embed_query(
                {"prompt": user_input, "model": model},
                host_ip=host_ip,
                store_in_db=False
            )
        except ValueError as e:
            return jsonify({"error": f"Failed to embed user input: {str(e)}"}), 500

        

        # Search embedded logs if enabled
        
        # Handle specific queries like current time or weather
        if re.search(r'\bcurrent time\b', user_input_lower):
            current_time = get_current_time()
            return jsonify({"response": current_time}), 200

        elif 'weather' in user_input_lower:
            location_match = re.search(r'weather in (\w+)', user_input_lower)
            location = location_match.group(1) if location_match else 'Denver'
            weather_info = get_weather(location)
            return jsonify({"response": weather_info}), 200

        #elif "workflow" in user_input_lower:
            #return jsonify({"response": "Would you like me to initiate the workflow? Click to proceed."}), 200

        # Handle general web search requests
        if any(keyword in user_input_lower for keyword in ['search for', 'find', 'look up']):
            search_query = extract_search_query(user_input)
            search_result = do_web_search(search_query)
            if search_result:
                return jsonify({"response": f"I found: {search_result}"}), 200
            else:
                return jsonify({"response": "Couldn't find any information on that topic."}), 200

        # Generate response from the model
        response = generate(
            user_input, model=model, host_ip=host_ip,
            num_predict=num_predict, temperature=temperature
        )
        logging.info(f"Model response: {response}")
        return jsonify({"response": response}), 200

    except Exception as e:
        logging.exception(f"An error occurred: {str(e)}")
        return jsonify({"error": "An unexpected error occurred."}), 500
 
# Logic to determine if Gabriel should ensure complete answers
def should_gemma_intervene(user_input, model_response):
    """
    Determine if Gemma should intervene in the conversation.
    """
    # Basic logic to decide if Gemma should intervene
    if model_response is None:
        logging.error("Model response is None in should_gemma_intervene.")
        return False
    
    if "stuck" in model_response.lower() or "not sure" in model_response.lower():
        logging.info("Gemma intervention triggered due to model uncertainty.")
        return True
    if "gemma" in model_response.lower() or "make it weird" in model_response.lower():
        logging.info("Gemma intervention triggered due to getting it.")
        return True
    if "?" in user_input.lower() or "what else can we do?" in user_input.lower():
        logging.info("Gemma intervention triggered due to user query.")
        return True
    return False
    
# Analyze log for relevance
def analyze_log_for_relevance(log_content, user_input):
    """
    Analyze the content of the log to extract information relevant to the current user input.
    """
    # Basic analysis to determine which parts of the log are relevant
    relevant_sentences = []
    user_keywords = user_input.lower().split()

    for line in log_content.splitlines():
        if any(keyword in line.lower() for keyword in user_keywords):
            relevant_sentences.append(line)

    if relevant_sentences:
        logging.debug(f"Relevant log information found: {' '.join(relevant_sentences)}")
        return ' '.join(relevant_sentences)
    else:
        logging.debug("No specific details were directly applicable, providing general context.")
        return "No specific details were directly applicable, but the context suggests that the issue might be related."

# Embedding query function
def embed_query(query, host_ip, store_in_db=True):
    """
    Sends the query to the embedding API and optionally stores embeddings in the DB.
    """
    embed_url = f'http://{host_ip}:11434/api/embed'
    headers = {'Content-Type': 'application/json'}

    payload = {
        "model": query.get("model", "Alex"),
        "input": query.get("prompt"),
        "truncate": True
    }

    try:
        logging.debug(f"Sending request to embedding API at {embed_url}")
        response = session.post(embed_url, headers=headers, json=payload, timeout=120)
        
        if response.status_code == 200:
            embed_response = response.json()
            if 'embeddings' in embed_response and len(embed_response['embeddings']) > 0:
                embeddings = np.array(embed_response['embeddings'])
                embeddings = ensure_2d(embeddings)

                if store_in_db:
                    logging.debug("Storing embedding in database.")
                    store_embedding(embeddings, query, host_ip)

                return embeddings
            else:
                logging.error("No embeddings generated for the given query.")
                raise ValueError("No embeddings generated for the given query.")
        else:
            logging.error(f"Embedding query failed with status code {response.status_code}: {response.text}")
            raise ValueError(f"Embedding query failed with status code {response.status_code}: {response.text}")
    except requests.RequestException as e:
        logging.error(f"Error connecting to Embedding API: {str(e)}")
        raise ValueError(f"Error connecting to Embedding API: {str(e)}")
        

# Store embedding in the database or locally
def store_embedding(embedding, query, host_ip=None):
    """
    Store the embedding vector and query information into PostgreSQL and also save locally.
    """
    #logging.info("Storing embedding locally.")
    #save_embedding_locally(embedding, query)

    # Generate a probable timestamp from the prompt
    timestamp = extract_timestamp_from_prompt(query.get('prompt', ''))

    # Store in PostgreSQL database
    conn = None
    try:
        conn = psycopg2.connect(f"dbname=chatbotdb user=chatbotuser password=changeme host={host_ip}")
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO embeddings (timestamp, prompt, embedding)
            VALUES (%s, %s, %s);
        """, (timestamp, query.get('prompt'), json.dumps(embedding.tolist())))

        conn.commit()
        logging.info("Embedding successfully stored in PostgreSQL.")
    except Exception as e:
        logging.error(f"Failed to store embedding in PostgreSQL: {str(e)}")
    finally:
        if conn:
            cursor.close()
            conn.close()
            
def save_embedding_locally(embedding, query, directory="local_embeddings"):
    """
    Save the embedding and query information to local storage as a JSON file.
    """
    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)

    # Generate a unique filename for each embedding using timestamp or query hash
    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
    file_name = f"embedding_{timestamp}.json"

    # Metadata to store along with the embedding
    metadata = {
        'timestamp': timestamp,
        'prompt': query.get('prompt', 'Unknown prompt'),
        'embedding': embedding.tolist()  # Convert numpy array to list for JSON serialization
    }

    # File path to save the embedding
    file_path = os.path.join(directory, file_name)

    # Save the metadata as JSON
    with open(file_path, 'w') as f:
        json.dump([metadata], f, indent=4)

    logging.debug(f"Stored embedding locally at {file_path}")

def generate(prompt, model="Alex", host_ip="10.79.85.40", template=None, context=None, stream=False, num_predict=-2, temperature=1, system=None):
    """
    Sends a request to the specified model.
    Modify behavior if Disco Biscuit mode is active.
    """
    global disco_biscuit_active

    if disco_biscuit_active:
        if model.lower() == "aqua" or model.lower() == "ai1":
            # Modify prompt or template for AI1 to act drunk
            prompt = f"You're feeling a little tipsy and carefree: {prompt}"
            template = "Drunk mode enabled"
        elif model.lower() == "gemma" or model.lower() == "ai2":
            # Modify prompt or template for AI2 to act like they are on THC
            prompt = f"You're super chill and seeing things differently: {prompt}"
            template = "THC mode enabled"

    url = f'http://{host_ip}:11434/api/generate'
    headers = {'Content-Type': 'application/json'}
    data = {
        "model": model,
        "prompt": prompt,
        "template": template,
        "system": system,
        "stream": stream,
        "context": context,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict
            },
        "save": model,
        "verbose": True,
        "num_predict": num_predict
    }

    try:
        logging.debug(f"Sending request to generate API at {url}")
        response = requests.post(url, headers=headers, json=data, timeout=120)
        if response.status_code == 200:
            response_data = response.json()
            return response_data.get('response', 'No response received')
        else:
            logging.error(f"Generate API returned status code: {response.status_code}")
            return None
    except requests.RequestException as e:
        logging.error(f"Error connecting to Generate API: {str(e)}")
        return None
        

# Function to search logs for relevant entries
def search_logs(model, query, brainbase):
    """
    Search for relevant log entries based on a query from the saved embedded logs.
    """
    embedded_files = []
    
    # Walk through the response directory to find embedded log files
    for root, dirs, files in os.walk(brainbase):
        for file in files:
            if file.endswith('.json'):
                embedded_files.append(os.path.join(root, file))

    logging.debug(f"Found {len(embedded_files)} files to search for query")

    # Loop through the files and look for matching logs
    for embedded_file in embedded_files:
        logging.debug(f"Checking file: {embedded_file}")
        try:
            with open(embedded_file, 'r') as f:
                data = json.load(f)
                #logging.debug(f"File {embedded_file} contents loaded. Data: {data}")
                
                if isinstance(data, dict):
                    data = [data]  # Wrap the dictionary in a list to process it as expected

                if isinstance(data, list):
                    for entry in data:
                        #logging.debug(f"Processing entry: {entry}")
                        if isinstance(entry, dict) and 'response' in entry:
                            response_data = entry['response']
                            #logging.debug(f"response_data content: {response_data}")
                            
                            if isinstance(response_data, dict):
                                # Log all fields in response_data for clarity
                                #logging.debug(f"Fields in response_data: {list(response_data.keys())}")
                                
                                # Search logic: check for query in 'model'
                                model_field = response_data.get('model', '')
                                logging.debug(f"Checking 'model' field in response_data: {model_field}")
                                
                                if 'model' in response_data and query.lower() in model_field.lower():
                                    logging.debug(f"Found matching log in {embedded_file}")
                                    return entry['response']
                            else:
                                logging.warning(f"Expected 'response' field to be a dict, got {type(response_data)} instead. ")
                        else:
                            logging.warning(f"Entry in {embedded_file} is not formatted as expected")
                else:
                    logging.warning(f"Unexpected data format in {embedded_file}. Expected a list, but got {type(data)}")

        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode JSON in {embedded_file}: {e}")
        except Exception as e:
            logging.error(f"Error reading or processing {embedded_file}: {str(e)}")

    logging.debug(f"No relevant logs found for query")
    return None  # Return None explicitly if no match found
    
# Ensure embeddings are in 2D format
def ensure_2d(embedding):
    """
    Ensures that the embeddings are 2D.
    """
    logging.debug(f"Original embedding shape: {embedding.shape}")
    if len(embedding.shape) == 1:
        logging.debug("Reshaping 1D array to 2D.")
        return embedding.reshape(1, -1)
    elif len(embedding.shape) == 3:
        logging.debug("Reshaping 3D array to 2D by collapsing.")
        return embedding.reshape(embedding.shape[0], -1)
    logging.debug("Embedding already 2D.")
    return embedding

# Improved search function using cosine similarity and retrieving multiple results
def search_embeddings_in_logs(query_embedding, host_ip=None, top_k=5, similarity_threshold=0.5):
    """
    Search for the most similar log entries in the database using cosine similarity.
    Returns the top-k matches above a similarity threshold.
    """
    conn = None
    results = []
    try:
        # Connect to the PostgreSQL database
        conn = psycopg2.connect(f"dbname=chatbotdb user=chatbotuser password=changeme host={host_ip}")
        cursor = conn.cursor()

        # Retrieve embeddings from the database
        cursor.execute("SELECT id, embedding FROM embeddings;")
        rows = cursor.fetchall()

        # Convert query embedding to 2D array for cosine similarity
        query_embedding = np.array(query_embedding).reshape(1, -1)

        # Store similarities for ranking
        similarities = []

        for row in rows:
            stored_embedding = np.array(row[1]).reshape(1, -1)

            # Check if shapes match before calculating similarity
            if stored_embedding.shape == query_embedding.shape:
                similarity = cosine_similarity(query_embedding, stored_embedding)[0][0]
                if similarity >= similarity_threshold:
                    similarities.append((similarity, row[0]))

        # Sort the results based on similarity score (highest first) and get the top K results
        similarities.sort(reverse=True, key=lambda x: x[0])
        top_matches = similarities[:top_k]

        # Retrieve the corresponding logs for the top matches
        for similarity, log_id in top_matches:
            log_content = retrieve_log_by_id(log_id, host_ip)
            if log_content:
                results.append({
                    "log_id": log_id,
                    "similarity": similarity,
                    "content": log_content
                })

        return results

    except Exception as e:
        logging.error(f"Error searching embeddings in logs: {str(e)}")
    finally:
        if conn:
            cursor.close()
            conn.close()

    return results

def retrieve_log_by_id(log_id, host_ip):
    """
    Retrieve log or prompt by embedding ID from the database.
    """
    try:
        conn = psycopg2.connect(f"host={host_ip} dbname=chatbotdb user=chatbotuser password=changeme")
        cursor = conn.cursor()
        
        cursor.execute("SELECT prompt FROM embeddings WHERE id = %s", (log_id,))
        log = cursor.fetchone()
        
        cursor.close()
        conn.close()

        if log:
            return log[0]  # Return the matched prompt or log
    except Exception as e:
        logging.error(f"Error retrieving log by ID: {str(e)}")

    return None

@app.route('/embed', methods=['POST'])
def embed():
    data = request.get_json()
    model = data.get("model", "Alex")
    input_data = data.get("prompt")  # Adjusted to match the API's expected input key
    truncate = data.get("truncate", True)

    # Convert model to lowercase for case-insensitive comparison
    model_lower = model.lower()

    # Determine host IP based on the model (case-insensitive)
    if model_lower == "aqua" or model_lower == "gabriel" or model_lower == "alex" or model_lower == "avery" or model_lower == "ai1":
        host_ip = "10.79.85.40"
    elif model_lower == "gemma" or model_lower == "gemma2" or model_lower == "ai2":
        host_ip = "10.79.85.47"
    else:
        return jsonify({"error": "Unknown model specified."}), 400


    # Construct the API URL using the selected host_ip
    test_url = f'http://{host_ip}:11434/api/embed'
    payload = {
        "model": model,
        "input": input_data,
        "truncate": truncate
    }

    try:
        response = requests.post(test_url, json=payload, timeout=120)

        if response.status_code == 200:
            embed_response = response.json()

            if 'embeddings' in embed_response:
                embeddings = np.array(embed_response['embeddings'])
                # Ensure 2D format for embeddings
                embeddings = ensure_2d(embeddings)
                flask_response = jsonify({"response": embeddings.tolist()})
            else:
                flask_response = jsonify({"error": "No embeddings found in response."}), 500
        else:
            flask_response = jsonify({"error": "Failed to fetch embedding."}), response.status_code

        flask_response.headers['Content-Type'] = 'application/json'  # Set header for Flask response
        return flask_response

    except Exception as e:
        flask_response = jsonify({"error": str(e)})
        flask_response.headers['Content-Type'] = 'application/json'
        return flask_response

def extract_timestamp_from_prompt(prompt):
    """
    Try to extract a probable timestamp from the given prompt string.
    If none is found, return the current timestamp.
    """
    # Try to find a date pattern in the prompt (e.g., "2024-10-09" or "October 9, 2024")
    date_patterns = [
        r'\b(\d{4}-\d{2}-\d{2})\b',          # YYYY-MM-DD format
        r'\b(\d{2}/\d{2}/\d{4})\b',           # MM/DD/YYYY format
        r'\b(\w+\s+\d{1,2},\s+\d{4})\b'       # Month Day, Year format (e.g., "October 9, 2024")
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, prompt)
        if match:
            try:
                # Try parsing the date into a datetime object
                timestamp = datetime.strptime(match.group(0), '%Y-%m-%d')
                return timestamp
            except ValueError:
                pass  # If parsing fails, continue

    # If no timestamp found in the prompt, use the current timestamp
    return datetime.now()

@app.route('/upload-log', methods=['POST'])
def upload_log():
    file = request.files.get('file')
    model = request.form.get('model', 'Alex')

    if not file:
        return jsonify({"error": "No file provided."}), 400

    file_path = os.path.join('/tmp', file.filename)
    file.save(file_path)

    try:
        from log_rag import process_file, extract_metadata, read_log_file

        # Run log_rag.py to process the file and extract messages
        output_messages = process_file(file_path, chunk_size=4096, model=model, use_embed=True)

        if output_messages is None:
            return jsonify({"error": "Failed to process the log file, no output messages were returned."}), 500

        # Read the file content for processing
        log_content = read_log_file(file_path)

        # Consolidate summaries
        chunk_summaries = {}
        for index, content in enumerate(output_messages):
            if content is None:
                continue

            # Extract metadata safely, ensuring content is valid
            try:
                metadata = extract_metadata(content, model=model, part=f"{index + 1}/{len(output_messages)}")
                chunk_id = f"chunk_{index + 1}"  # Use a unique identifier for each chunk
                
                # Consolidate all related summaries for each chunk
                if chunk_id not in chunk_summaries:
                    chunk_summaries[chunk_id] = {
                        "model": model,
                        "file": file.filename,
                        "part": f"{index + 1}/{len(output_messages)}",
                        "summaries": []
                    }
                
                chunk_summaries[chunk_id]["summaries"].append(metadata.get('summary', 'No summary available'))

            except Exception as e:
                logging.error(f"Failed to extract metadata for part {index + 1}: {str(e)}")
                if f"chunk_{index + 1}" not in chunk_summaries:
                    chunk_summaries[f"chunk_{index + 1}"] = {
                        "model": model,
                        "file": file.filename,
                        "part": f"{index + 1}/{len(output_messages)}",
                        "summaries": [f"Failed to extract metadata: {str(e)}"]
                    }

        # Create a consolidated response
        consolidated_summaries = []
        for chunk_id, data in chunk_summaries.items():
            consolidated_summary = " ".join(data["summaries"])
            consolidated_summaries.append({
                "model": data["model"],
                "file": data["file"],
                "part": data["part"],
                "summary": consolidated_summary
            })

        return jsonify({"summaries": consolidated_summaries}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        os.remove(file_path)


@app.route('/embed-content', methods=['POST'])
def embed_content():
    # Get the user input for the URL and model
    file = request.files.get('file')
    github_url = request.form.get('github')
    confluence_url = request.form.get('confluence')
    org_name = request.form.get('org_name')
    ssh_repo = request.form.get('ssh_repo')
    ssh_url = request.form.get('ssh_url')
    public_github_url = request.form.get('public_github')
    generic_url = request.form.get('generic_url')
    space_key = request.form.get('space_key', 'CX')
    title = request.form.get('title', 'default')
    model = request.form.get('model', 'Alex')
    host_ip = request.form.get('host_ip', '10.79.85.40')
    username = request.form.get('username')
    password = request.form.get('password')

    # Determine which tool to use based on the URL provided
    tool = None
    if generic_url:
        if 'bash' in generic_url:
            tool = 'bash'
            tool_data = {"command": request.form.get("command")}
        if 'api_call' in generic_url:
            tool = 'api_call'
            tool_data = {
                "url": request.form.get("api_url"),
                "method": request.form.get("http_method"),
                "headers": request.form.get("headers", {}),
                "data": request.form.get("data", {})
            }

    if tool:
        try:
            # Internal API call to the respective tool route
            tool_endpoint = f'http://localhost:5000/tools/{tool}'
            response = requests.post(tool_endpoint, json=tool_data)

            # Handle the response from the tool route
            if response.status_code == 200:
                result = response.json()
                return jsonify({"result": f"{tool.capitalize()} tool executed successfully.", "data": result}), 200
            else:
                return jsonify({"error": f"Failed to execute {tool} tool.", "details": response.text}), response.status_code

        except requests.exceptions.RequestException as e:
            return jsonify({"error": f"Request to internal tool endpoint failed: {str(e)}"}), 500


    # Handle public GitHub repository cloning
    if public_github_url:
        repo_path = clone_or_pull_github_repo(public_github_url)
        if repo_path:
            content = extract_github_content(repo_path)
            for item in content:
                metadata = {
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'error_type': 'N/A',
                    'summary': f"Content from file: {item['file_path']} in repository {public_github_url}",
                    'part': 'GitHub'
                }
                success = embed_and_store_content(item['content'], metadata)
                if not success:
                    return {"error": f"Failed to embed content from {item['file_path']}"}, 500
        return {"result": "Public GitHub content successfully embedded."}, 200


    # Handle GitHub HTTPS URL
    if github_url:
        repo_path = clone_or_pull_github_repo(github_url)
        if repo_path:
            content = extract_github_content(repo_path)
            for item in content:
                metadata = {
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'error_type': 'N/A',
                    'summary': f"Content from file: {item['file_path']}",
                    'part': 'GitHub'
                }
                success = embed_and_store_content(item['content'], metadata)
                if not success:
                    return {"error": f"Failed to embed content from {item['file_path']}"}, 500

        return {"result": "GitHub HTTPS content successfully embedded."}, 200

    # Handle GitHub SSH URL
    if ssh_url:
        ssh_key_path = os.getenv('SSH_KEY_PATH')  # Assuming SSH key path is stored as an environment variable

        try:
            # Attempt to clone using SSH
            logging.info(f"Attempting to clone repository using SSH: {ssh_url}")
            clone_command = ["ssh-agent", "bash", "-c", f"ssh-add {ssh_key_path}; git clone {ssh_url}"]
            result = subprocess.run(clone_command, capture_output=True, text=True, shell=True)

            if result.returncode != 0:
                logging.error(f"Failed to clone repository using SSH: {result.stderr}")
                return {"error": f"Failed to clone repository using SSH: {result.stderr}"}, 500

            repo_name = ssh_url.split('/')[-1].replace('.git', '')
            repo_path = os.path.join('repositories', repo_name)
            if os.path.exists(repo_path):
                content = extract_github_content(repo_path)
                for item in content:
                    metadata = {
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'error_type': 'N/A',
                        'summary': f"Content from file: {item['file_path']} in SSH repository",
                        'part': 'GitHub SSH'
                    }
                    success = embed_and_store_content(item['content'], metadata)
                    if not success:
                        return {"error": f"Failed to embed content from {item['file_path']}"}, 500

            return {"result": "GitHub SSH content successfully embedded."}, 200

        except Exception as e:
            logging.error(f"Exception while cloning repositories using SSH: {str(e)}")
            return {"error": f"Exception occurred: {str(e)}"}, 500

    # Handle Confluence URL if provided
    if confluence_url:
        # Extract the space key from the Confluence URL
        space_key = extract_space_key_from_url(confluence_url)
        
        if not space_key:
            return jsonify({"error": "Failed to extract space key from URL."}), 400

        # Fetch and embed Confluence content using the extracted space key
        confluence_content = fetch_confluence_page(space_key, title, confluence_url, pat_token)
        if confluence_content:
            metadata = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'error_type': 'N/A',
                'summary': f"Content from Confluence page: {title}",
                'part': 'Confluence'
            }
            success = embed_and_store_content(confluence_content, metadata, host_ip=host_ip)
            if success:
                return jsonify({"result": "Confluence content successfully embedded."}), 200
            else:
                return jsonify({"error": "Failed to embed Confluence content."}), 500
        else:
            return jsonify({"error": "Failed to fetch Confluence content."}), 500

    if not generic_url:
        return jsonify({"error": "No URL provided."}), 400

    session = requests.Session()  # Using a session to persist cookies

    try:
        if 'dishanywhere.com' in generic_url:
            # Login URL for CAS
            login_url = "https://cas-cha01.dishanywhere.com:8443/cas/login"

            # Access the CAS login page
            login_page = session.get(login_url, verify=False)
            if login_page.status_code != 200:
                return jsonify({"error": "Failed to access login page."}), 500

            # Parse login page for the execution token
            soup = BeautifulSoup(login_page.text, 'html.parser')
            execution_token = soup.find('input', {'name': 'execution'})['value']

            # Prepare login payload
            payload = {
                'username': username,
                'password': password,
                'execution': execution_token,
                '_eventId': 'submit',
                'geolocation': ''
            }

            # Attempt login
            login_response = session.post(login_url, data=payload, verify=False, timeout=30)

            if login_response.status_code != 200 or "Incorrect username or password" in login_response.text:
                return jsonify({"error": "Login failed. Incorrect credentials.", "needs_credentials": True}), 401

            # Access the generic URL after login
            response = session.get(generic_url, verify=False, timeout=30)
        else:
            # If no login is required, attempt direct access
            response = session.get(generic_url, verify=False, timeout=30)

        response.raise_for_status()
        page_content = response.text

        # Extract meaningful text using BeautifulSoup
        soup = BeautifulSoup(page_content, 'html.parser')
        clean_content = soup.get_text(separator="\n")

        metadata = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'error_type': 'N/A',
            'summary': f"Content from webpage: {generic_url}",
            'part': 'Generic URL'
        }

        success = embed_and_store_content(clean_content, metadata)
        if success:
            return jsonify({"result": "Generic URL content successfully embedded."}), 200
        else:
            return jsonify({"error": "Failed to embed generic URL content."}), 500

    except requests.exceptions.SSLError as e:
        return jsonify({"error": f"SSL Error: {str(e)}"}), 500
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to fetch content from the provided URL: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/disco_biscuit', methods=['POST'])
def disco_biscuit():
    """
    Temporarily changes AI1 to act drunk and AI2 to act like they are on THC for 15 minutes.
    """
    global disco_biscuit_active
    if disco_biscuit_active:
        return jsonify({"message": "Disco Biscuit mode is already active!"}), 400

    disco_biscuit_active = True
    logging.info("Disco Biscuit mode activated: AI1 is drunk, AI2 is on THC.")

    # Schedule to reset behavior after 15 minutes
    Timer(5 * 60, reset_disco_biscuit).start()

    return jsonify({"message": "Disco Biscuit mode activated for 15 minutes."}), 200


def reset_disco_biscuit():
    """
    Resets the Disco Biscuit behavior for AI1 and AI2.
    """
    global disco_biscuit_active
    disco_biscuit_active = False
    logging.info("Disco Biscuit mode deactivated: AI1 and AI2 returning to normal behavior.")

@app.route('/disco_biscuit_status', methods=['GET'])
def disco_biscuit_status():
    """
    Check the status of Disco Biscuit mode.
    """
    return jsonify({"active": disco_biscuit_active}), 200

@app.route('/tools/<tool>', methods=['GET', 'POST'])
def tools(tool):
    if tool == 'allowed':
        try:
            with open('tools/bash.txt', 'r') as f:
                allowed_commands = f.read()
            # Return the contents as plain text
            return Response(allowed_commands, mimetype='text/plain')
        except Exception as e:
            return jsonify({'error': f'Failed to read allowed commands: {str(e)}'}), 500

    if request.method == 'POST':
        data = request.get_json()
        argument = data.get('argument')

        if tool == 'ping':
            ip_address = argument
            result = ping_ip(ip_address)
            return jsonify({'result': result})

        elif tool == 'bash':
            command = argument
            result = run_bash_command(command)
            return jsonify({'result': result})

        elif tool == 'fart':
            # Assuming dart is another tool that requires specific processing
            result = run_dart_command(argument)  # Example function
            return jsonify({'result': result})

        else:
            return jsonify({'error': 'Invalid tool requested'}), 400
    else:
        return jsonify({'error': 'Invalid request method'}), 405

def ping_ip(ip_address):
    """Ping an IP address for both Linux and Windows."""
    try:
        # Check operating system
        os_name = platform.system().lower()

        # Use appropriate command based on OS
        if 'windows' in os_name:
            command = ['ping', '-n', '4', ip_address]
        else:
            command = ['ping', '-c', '4', ip_address]

        result = subprocess.run(command, capture_output=True, text=True)
        return result.stdout if result.returncode == 0 else result.stderr
    except Exception as e:
        return f"Error pinging IP address: {str(e)}"

@app.route('/run-command', methods=['POST'])
def run_command():
    data = request.get_json()
    command = data.get('command')
    content = data.get('content')

    if command and content:
        try:
            # Create and write to the file using nano
            with open(command.split(' ')[1], 'w') as f:
                f.write(content)
            return jsonify({"success": True, "message": f"{command} executed successfully."})
        except Exception as e:
            return jsonify({"success": False, "message": str(e)})
    else:
        return jsonify({"success": False, "message": "Invalid command or content."})
        

def run_bash_command(command):
    """Run a bash command for Linux and Windows environments."""
    try:
        os_name = platform.system().lower()
        
        if 'windows' in os_name:
            # For Windows, use PowerShell or cmd
            result = subprocess.run(['powershell', '-Command', command], capture_output=True, text=True, shell=True)
        else:
            # For Linux, use bash directly
            result = subprocess.run(command, capture_output=True, text=True, shell=True)

        return result.stdout if result.returncode == 0 else result.stderr
    except Exception as e:
        return f"Error running bash command: {str(e)}"

def run_dart_command(argument):
    """Run a dart command, assuming it's installed in the system."""
    try:
        os_name = platform.system().lower()

        # Generic example to simulate running a Dart command
        if 'windows' in os_name:
            command = ['dart', 'run', argument]
        else:
            command = ['dart', argument]

        result = subprocess.run(command, capture_output=True, text=True)
        return result.stdout if result.returncode == 0 else result.stderr
    except Exception as e:
        return f"Error running Dart command: {str(e)}"
        
@app.route('/trigger-workflow', methods=['POST'])
def trigger_workflow():
    """
    Trigger the process where JAMbot generates instructions, gathers resources,
    runs commands, and embeds content.
    """
    data = request.get_json()
    task_description = data.get("task_description", "Generate data and predictions.")
    original_request = data.get("original_request")

    try:
        # Step 1: Write instructions
        filename = write_instructions(task_description)

        # Step 2: Generate resource list based on instructions
        resource_filename = generate_resource_list(filename)

        # Step 3: Run commands and store results
        data_filename = run_commands(resource_filename)

        # Step 4: Process the data for embedding and conversation handling
        process_data_for_embedding(data_filename, original_request)

        return jsonify({"response": f"Workflow completed successfully. Data stored in {data_filename}."}), 200

    except Exception as e:
        logging.error(f"Workflow failed: {e}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

def write_instructions(task_description):
    filename = generate_three_word_name() + ".instruct"
    path = f"instructions/{filename}"
    with open(path, 'w') as file:
        file.write(task_description)
    return filename

def generate_three_word_name():
    import random
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]
    #return '-'.join(random.sample(words, 3))
    return "workflow"
    
def generate_resource_list(filename):
    # Read the content of the task file
    with open(f"instructions/{filename}", 'r') as file:
        task = file.read()

    # Prepare a prompt asking the bot to extract tools/resources
    prompt = f"""
    The following task requires various tools or resources to complete. 
    Please extract and list each tool or resource needed to accomplish the task described below:
    
    Task:
    {task}

    Please respond with a CSV-style output. Each line should contain:
    Tool/Resource Name, Specific information needed, Required Bash Command (if applicable), the url to find the information.
    """

    # Send the prompt to the model using the `generate()` function
    try:
        response = generate(prompt, model="Alex", num_predict=-2)
        resources = response.strip().split('\n')
    except Exception as e:
        print(f"Error generating resource list: {e}")
        return None

    # Create the resource filename by replacing the extension
    resource_filename = filename.replace('.instruct', '.resources')

    # Save the extracted resources to the new file
    with open(f"instructions/{resource_filename}", 'w') as file:
        for resource in resources:
            file.write(resource + '\n')

    return resource_filename

def run_commands(resource_filename):
    data_filename = resource_filename.replace('.resources', '.data')

    # Collect all outputs for handle_conversation()
    all_results = []

    with open(f"instructions/{resource_filename}", 'r') as file:
        commands = [line.strip().split(',') for line in file.readlines()]

    with open(f"instructions/{data_filename}", 'a') as data_file:
        for command_info in commands:
            if len(command_info) < 1:
                print(f"Skipping malformed line: {command_info}")
                continue  # Skip empty or malformed lines

            # Extract command or URL from the last element of each line
            target = command_info[-1].strip()

            if target.startswith("http"):
                # Handle URL
                print(f"Fetching data from URL: {target}")
                try:
                    response = requests.get(target)
                    response.raise_for_status()  # Raise an error for bad responses
                    result = response.text
                except requests.RequestException as e:
                    result = f"Failed to fetch URL {target}: {str(e)}"
            else:
                # Handle bash command
                print(f"Executing command: {target}")
                result = subprocess.getoutput(target)

            # Save result to the .data file
            data_file.write(f"Target: {target}\n")
            data_file.write(f"Result:\n{result}\n\n")

            # Collect result for handle_conversation()
            all_results.append(f"Target: {target}\nResult:\n{result}")

    # Send all results to handle_conversation() via HTTP POST
    try:
        conversation_payload = {
            "prompt": "Workflow completed. Here are the results:",
            "data": "\n\n".join(all_results)
        }

        response = requests.post('http://localhost:5000/ollama', json=conversation_payload)
        if response.status_code == 200:
            print("Conversation handled successfully.")
        else:
            print(f"Failed to handle conversation: {response.text}")
    except Exception as e:
        print(f"Error handling conversation: {str(e)}")

    return data_filename
    
    
def process_data_for_embedding(data_filename, original_request):
    with open(f"instructions/{data_filename}", 'r') as file:
        data_content = file.read()

    # Explanation for the bot to understand where it is in the workflow
    prompt = f"""
    You are in the middle of executing a complex workflow. The original task requested is as follows:

    Original Task:
    {original_request}

    Below is the data that has been gathered so far. Please process this data and complete as much of the original task as possible. 
    If there are still missing resources or incomplete tasks, notify the system to continue gathering more information.

    Gathered Data:
    {data_content}
    """

    # Prepare the payload for embedding
    embed_payload = {
        'generic_url': data_content,
        'model': 'Alex',
        'title': 'Data Summary',
        'space_key': 'CX',
        'host_ip': '10.79.85.40'
    }

    # Embed the content via POST request
    try:
        embed_response = requests.post('http://localhost:5000/embed-content', data=embed_payload)
        if embed_response.status_code != 200:
            print(f"Failed to embed content: {embed_response.text}")
            return
    except Exception as e:
        print(f"Error during embedding: {str(e)}")
        return

    # Prepare the conversation payload to send to handle_conversation()
    conversation_payload = {
        "prompt": prompt,
        "data": data_content
    }

    # Handle conversation by sending the data to the chatbot
    try:
        conversation_response = requests.post('http://localhost:5000/ollama', json=conversation_payload)
        if conversation_response.status_code == 200:
            print("Conversation handled successfully.")
        else:
            print(f"Failed to handle conversation: {conversation_response.text}")
    except Exception as e:
        print(f"Error handling conversation: {str(e)}")
        return

    # Restart the workflow if the .resources file still exists
    resource_filename = data_filename.replace('.data', '.resources')
    try:
        with open(f"instructions/{resource_filename}", 'r') as file:
            # If the resources file has content, restart the workflow
            print("Resources still available. Restarting workflow...")
            trigger_workflow()
    except FileNotFoundError:
        print("No more resources to process. Workflow completed.")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
