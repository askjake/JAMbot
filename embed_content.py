import os
import json
import logging
import argparse
from collections import defaultdict
from drain3 import TemplateMiner
import requests
import re
import tarfile
import urllib.parse
import numpy as np
import psycopg2
import hashlib
import base64
import time
import git
from atlassian import Confluence
from requests.exceptions import RequestException
import getpass
from bs4 import BeautifulSoup
from requests.auth import HTTPBasicAuth

# Set up logging configuration
logging.basicConfig(filename='logJAM.txt', level=logging.DEBUG)
logger = logging.getLogger()

# Global settings
credentials_file = 'credentials.txt'
response_directory = 'ai_responses'
interim_directory = os.path.join(response_directory, 'brainbase')
brainbase = os.path.join(response_directory, 'brainbase')  
brainbed = os.path.join(response_directory, 'brainbed')  
embed_url = 'http://localhost:5000/embed'
git_token = 'SHA256:7i0WMHqBH+Qq8KK/sDshBIQ9/98o1yHpYGLfX00pMUM'


confluence_api_url = "https://confluence.dtc.dish.corp/rest/api/content"
confluence_token = 'MzQxMjI2OTk5OTAwOrH7F6ttTdr4BKYL02OpL6f+EU7I'
pat_token  = 'MzQxMjI2OTk5OTAwOrH7F6ttTdr4BKYL02OpL6f+EU7I'
username = 'montjac'
password = 'Chang3m3!'
cert = 'JAMbot.cert'

def setup_logging(verbose=True):
    """
    Configure logging to include console output if verbose is True.
    """
    if verbose:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
def extract_org_name_from_url(org_url):
    """
    Extract the organization name from the given GitHub URL.
    """
    # Match URLs like: https://git.dtc.dish.corp/orgs/DT-ENG or similar patterns
    match = re.search(r'https:\/\/git\.dtc\.dish\.corp\/(?:orgs\/|groups\/|)[\w-]+\/([\w-]+)', org_url)
    if match:
        return match.group(1)
    else:
        return None


org_url = f'https://{git_token}git.dtc.dish.corp/orgs/DT-ENG'
ssh_key_path = os.getenv('SSH_KEY_PATH', '~/.ssh/id_rsa')  # Adjust accordingly
repositories_folder = 'repositories'

def list_repositories(org_url, git_token):
    """
    Fetch all repositories from the given organization URL.
    """
    headers = {
        "Private-Token": git_token
    }
    repos = []
    page = 1

    while True:
        try:
            response = requests.get(f"{org_url}/projects", headers=headers, params={'page': page, 'per_page': 100}, verify=False)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                break

            repos.extend(data)
            page += 1
        except requests.RequestException as e:
            logger.error(f"Error fetching repositories: {str(e)}")
            break

    return repos


def clone_repository(repo_url, repo_name):
    """
    Clone a given repository using Git.
    """
    repo_path = os.path.join(repositories_folder, repo_name)

    if os.path.exists(repo_path):
        logger.info(f"Repository {repo_name} already exists, pulling latest changes.")
        try:
            repo = git.Repo(repo_path)
            repo.remotes.origin.pull()
        except Exception as e:
            logger.error(f"Failed to pull repository {repo_name}: {str(e)}")
    else:
        logger.info(f"Cloning repository: {repo_name}")
        try:
            # SSH cloning example, switch to https if using token.
            clone_command = f"git clone git@git.dtc.dish.corp:{repo_url} {repo_path}"
            os.system(clone_command)
        except Exception as e:
            logger.error(f"Failed to clone repository {repo_name}: {str(e)}")


def fetch_repositories(org_url, git_token, username, password):
    """
    Fetch all repositories from a GitHub organization. If login is required, prompt the user for credentials.
    """
    org_name = extract_org_name_from_url(org_url)
    if not org_name:
        logger.error("Invalid organization URL provided.")
        return []

    url = f"https://git.dtc.dish.corp/api/v4/groups/{org_name}/projects"
    headers = {"Private-Token": git_token}
    repos = []
    page = 1

    logger.info(f"Starting to fetch repositories for organization: {org_name}")

    session = requests.Session()  # Using a session to persist credentials

    while True:
        try:
            response = session.get(url, headers=headers, params={'page': page, 'per_page': 100}, verify=False)

            # Check if we are being redirected to a login page
            if response.status_code == 302 or '/login' in response.url:
                logger.info("Login is required to access this resource.")
                
                # Perform GET request to obtain the login page
                login_page_response = session.get(response.url, verify=False)
                soup = BeautifulSoup(login_page_response.text, 'html.parser')

                # Check if there's an authenticity token (CSRF) needed for login
                authenticity_token = soup.find('input', {'name': 'authenticity_token'})
                if authenticity_token:
                    authenticity_token = authenticity_token.get('value')

                # Prepare login payload including the authenticity token if present
                login_payload = {
                    'username': username,
                    'password': password,
                }

                if authenticity_token:
                    login_payload['authenticity_token'] = authenticity_token

                # Log in using POST
                login_url = "https://git.dtc.dish.corp/users/sign_in"
                login_response = session.post(login_url, data=login_payload, verify=False)

                # Check if login was successful
                if login_response.status_code == 200 and "sign_out" in login_response.text:
                    logger.info("Login successful.")
                    # Retry fetching the repositories
                    response = session.get(url, headers=headers, params={'page': page, 'per_page': 100}, verify=False)
                else:
                    logger.error(f"Login failed. username: {username}, password: {password}")
                    return []

            # If we have a successful response, continue processing the repositories
            response.raise_for_status()
            data = response.json()

            if not data:
                logger.info(f"No more repositories found at page {page}. Ending fetch.")
                break

            repos.extend(data)
            logger.info(f"Fetched page {page}, fetched {len(data)} repositories.")
            page += 1

        except RequestException as e:
            logger.error(f"Error fetching repositories: {str(e)}")
            break

    logger.info(f"Total repositories fetched: {len(repos)}")
    return repos
    
# Example usage
git_token = 'SHA256:7i0WMHqBH+Qq8KK/sDshBIQ9/98o1yHpYGLfX00pMUM'
org_url = f'https://{git_token}git.dtc.dish.corp/orgs/DT-ENG'

# Replace with your credentials

repositories = fetch_repositories(org_url, git_token, username, password)

if repositories:
    logger.info(f"Repositories successfully fetched: {len(repositories)}")
else:
    logger.error("Failed to fetch repositories.")
    
def fetch_repositories_ssh(org_name, ssh_key_path=None):
    """
    Fetch repositories from a GitHub organization using SSH.
    """
    url = f"git@git.dtc.dish.corp:{org_name}.git"
    repos = []

    try:
        logger.info(f"Attempting to clone repositories for organization: {org_name} using SSH.")
        
        clone_command = ["GIT_SSH_COMMAND='ssh -i /path/to/your_private_key'", "git", "clone", "--mirror", url]
        if ssh_key_path:
            # Use custom SSH key if provided
            clone_command = [
                "ssh-agent", "bash", "-c",
                f"ssh-add {ssh_key_path}; " + " ".join(clone_command)
            ]
        
        result = subprocess.run(clone_command, capture_output=True, text=True)

        if result.returncode == 0:
            logger.info(f"Successfully cloned repository for organization: {org_name}")
            repos.append(url)  # Assuming the repository URL as the identifier
        else:
            logger.error(f"Failed to clone repository: {result.stderr}")

    except Exception as e:
        logger.error(f"Exception while cloning repositories: {str(e)}")

    logger.info(f"Total repositories fetched: {len(repos)}")
    return repos

def process_organization_repos(org_name, git_token, local_path='repositories'):
    """
    Process all repositories under an organization and embed their content.
    """
    repos = fetch_repositories(org_name, git_token)
    if not repos:
        logging.info("No repositories found or failed to fetch.")
        return
    
    for repo in repos:
        repo_url = repo.get("ssh_url_to_repo")  # or "http_url_to_repo" for HTTPS cloning
        repo_name = repo.get("name")
        if repo_url:
            logging.info(f"Processing repository: {repo_name}")
            repo_path = clone_or_pull_github_repo(repo_url, local_path)
            if repo_path:
                # Extract and embed content
                content = extract_github_content(repo_path)
                for item in content:
                    logger.info(f"Content snippet from {item['file_path']}: {item['content'][:100]}")  # Log first 100 chars
                    metadata = {
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'error_type': 'N/A',
                        'summary': f"Content from file: {item['file_path']} in repository {repo_name}",
                        'part': 'GitHub'
                    }
                    embed_and_store_content(item['content'], metadata)
        
        
# Function to clone or pull a GitHub repository
def clone_or_pull_github_repo(github_url, local_path='repositories'):
    try:
        os.makedirs(local_path, exist_ok=True)
        repo_name = github_url.split('/')[-1].replace('.git', '')
        repo_path = os.path.join(local_path, repo_name)

        if os.path.exists(repo_path):
            logger.info(f"Pulling latest changes for repository: {repo_name}")
            repo = git.Repo(repo_path)
            repo.remotes.origin.pull()
        else:
            logger.info(f"Cloning repository: {repo_name}")
            git.Repo.clone_from(github_url, repo_path)
        return repo_path

    except Exception as e:
        logger.error(f"Failed to clone or pull GitHub repository: {str(e)}")
        return None
        

# Function to extract content from a GitHub repository
def extract_github_content(repo_path):
    content = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith('.py') or file.endswith('.md') or file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    file_content = f.read()
                    content.append({"file_path": file_path, "content": file_content})
                    logger.info(f"Extracted content from file: {file_path}, length: {len(file_content)}")
    return content
    
def convert_view_url_to_api_url(view_url):
    """
    Converts a Confluence viewpage URL into the corresponding REST API URL.
    """
    try:
        # Extract spaceKey and title from the view URL
        match = re.search(r'spaceKey=([\w-]+)&title=([\w+%-]+)', view_url)
        if not match:
            logger.error("Invalid Confluence URL format.")
            return None
        
        space_key = match.group(1)
        title = match.group(2)

        # Decode the title if it has special characters
        decoded_title = urllib.parse.unquote_plus(title)
        title_encoded = urllib.parse.quote(title)


        # Construct the API URL
        params = {
            "type": "page",
            "spaceKey": space_key,
            "title": title_encoded,
            "expand": "body.storage"
        }

        api_url = f"{confluence_api_url}?{urllib.parse.urlencode(params)}"
        logger.debug(f"Converted API URL: {api_url}")
        return api_url

    except Exception as e:
        logger.error(f"Error converting view URL: {str(e)}")
        return None        
        
def list_confluence_spaces(confluence_url, token):
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    api_url = f"{confluence_url}/rest/api/space"
    logger.info(f"listing spaces: {api_url}")
    try:
        response = requests.get(api_url, headers=headers, verify=False)
        response.raise_for_status()
        spaces_data = response.json()
        if 'results' in spaces_data:
            spaces = [space['key'] for space in spaces_data['results']]
            logger.info(f"Available spaces: {spaces}")
            return spaces
    except requests.exceptions.RequestException as e:
        logger.error(f"Error listing spaces: {str(e)}")
        return []


# Fetch a Confluence page content
def fetch_confluence_page(space_key, title, confluence_url, token):
    """
    Fetch a Confluence page content by space key and title.
    """
    # Encode the title for URL safety
    title_encoded = urllib.parse.quote_plus(title)
    base_url = strip_to_base_url(confluence_url)

    if not base_url:
        logger.error("Failed to determine base URL.")
        return None

    # Construct the correct API URL
    api_url = f"{base_url}/rest/api/content?type=page&spaceKey={space_key}&title={title_encoded}&expand=body.storage"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    try:
        logger.debug(f"Fetching Confluence page with API URL: {api_url}")
        response = requests.get(api_url, headers=headers, verify=False)
        response.raise_for_status()
        data = response.json()

        if data and 'results' in data and len(data['results']) > 0:
            page_content = data['results'][0]['body']['storage']['value']
            logger.info(f"Successfully fetched content: {page_content[:100]}...")
            return page_content
        else:
            logger.error("Confluence page not found for the given API URL.")
            return None

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching Confluence page: {str(e)}")
        return None


# Function to generate a unique hash for content
def generate_content_hash(content):
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

# Function to check if the content has already been embedded
def content_already_embedded(content_hash, host_ip='10.79.85.40'):
    conn = None
    try:
        conn = psycopg2.connect(f"dbname=chatbotdb user=chatbotuser password=changeme host={host_ip}")
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM embeddings WHERE content_hash = %s", (content_hash,))
        exists = cursor.fetchone() is not None
        return exists
    except Exception as e:
        logger.error(f"Error checking if content is already embedded: {str(e)}")
        return False
    finally:
        if conn:
            cursor.close()
            conn.close()

def split_content_into_chunks(content, max_length=3000, overlap=200):
    """
    Split the content into overlapping chunks of a specified length.
    Each chunk will have an overlap with the previous one.
    
    Args:
        content (str): The content to be split.
        max_length (int): The maximum length of each chunk.
        overlap (int): The number of overlapping characters between consecutive chunks.
        
    Returns:
        list: A list of content chunks.
    """
    content_chunks = []
    start = 0

    while start < len(content):
        end = min(start + max_length, len(content))
        content_chunks.append(content[start:end])
        
        if end == len(content):  # No more content left
            break
        
        # Move the start pointer forward, with overlap
        start = end - overlap

    return content_chunks

# Updated Recursive Function to Fetch and Embed Confluence Pages
def fetch_and_embed_confluence_page(confluence, space_key, title, visited_pages, metadata):
    logger.info(f"Fetching Confluence page: {space_key}/{title}")
    try:
        # Fetch the Confluence page content using its title and space key
        page = confluence.get_page_by_title(space=space_key, title=title, expand='body.storage')
        if not page or 'body' not in page or 'storage' not in page['body']:
            logger.error(f"Failed to fetch Confluence page: {space_key}/{title}")
            return False

        content = page['body']['storage']['value']
        if not content:
            logger.error(f"Empty content found for Confluence page: {space_key}/{title}")
            return False

        content_hash = generate_content_hash(content)
        if content_hash in visited_pages:
            logger.info(f"Already embedded content for page: {space_key}/{title}, skipping.")
            return True

        visited_pages.add(content_hash)
        
        # Prepare metadata
        metadata['summary'] = f"Content from Confluence page: {title}"
        metadata['part'] = 'Confluence'

        # Embed and store content
        embed_and_store_content(content, metadata)

        # Extract links to other Confluence pages within the current page
        soup = BeautifulSoup(content, 'html.parser')
        links = soup.find_all('a', href=True)

        for link in links:
            href = link['href']
            if '/pages/viewpage.action' in href:
                match = re.search(r'spaceKey=([\w-]+)&title=([\w+-]+)', href)
                if match:
                    linked_space_key, linked_title = match.groups()
                    linked_title = linked_title.replace('+', ' ')
                    # Recursively embed the linked page
                    fetch_and_embed_confluence_page(confluence, linked_space_key, linked_title, visited_pages, metadata)

        return True

    except RequestException as e:
        logger.error(f"Error fetching Confluence page: {str(e)}")
        return False
       
def strip_to_base_url(confluence_url):
    """
    Strip any path after the domain to ensure we have the base URL.
    """
    match = re.match(r'(https?://[^/]+)', confluence_url)
    if match:
        return match.group(1)
    else:
        logger.error(f"Invalid Confluence URL format: {confluence_url}")
        return None

def get_confluence_spaces(confluence_url, token):
    # Strip down to the base URL to construct the API URL properly
    base_url = strip_to_base_url(confluence_url)
    if not base_url:
        logger.error("Failed to determine base URL.")
        return []

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    url = f"{base_url}/rest/api/space"
    try:
        response = requests.get(url, headers=headers, verify=False)  # Verify can be True in production
        response.raise_for_status()
        spaces_data = response.json()
        if 'results' in spaces_data:
            spaces = {space['name']: space['key'] for space in spaces_data['results']}
            logger.debug(f"Available spaces: {spaces}")
            return spaces
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching spaces: {str(e)}")
    
    return []
   
   # Function to embed and store content
def embed_and_store_content(content, metadata, host_ip='10.79.85.40', max_chunk_length=3000, overlap=200):
    if not content:
        logger.error("Empty content passed to embedding function. Skipping.")
        return False

    # Split content into chunks
    content_chunks = split_content_into_chunks(content, max_length=max_chunk_length, overlap=overlap)
    logger.info(f"Split content into {len(content_chunks)} chunks, each up to {max_chunk_length} characters with {overlap} overlap.")

    success = True
    for index, chunk in enumerate(content_chunks):
        logger.info(f"Embedding chunk {index + 1}/{len(content_chunks)} with metadata: {metadata}")

        # Generate a unique hash for the current chunk
        content_hash = generate_content_hash(chunk)
        if content_already_embedded(content_hash, host_ip=host_ip):
            logger.info(f"Chunk {index + 1} already embedded. Skipping.")
            continue

        # Create payload for embedding
        payload = {
            "model": "Alex",
            "prompt": chunk,
            "truncate": True
        }

        try:
            response = requests.post(embed_url, json=payload)
            response.raise_for_status()

            # Extract embeddings from response
            embeddings = response.json().get('response')
            if embeddings is None:
                logger.error(f"Embedding response is None for chunk {index + 1}. Skipping.")
                success = False
                continue

            logger.info(f"Successfully received embedding response of length: {len(embeddings)} for chunk {index + 1}.")
            store_embedding(embeddings, metadata, content_hash)
            logger.info(f"Successfully stored embedding and metadata for chunk {index + 1}.")

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to embed chunk {index + 1}: {str(e)}")
            success = False
        except TypeError as e:
            logger.error(f"Failed to store embeddings for chunk {index + 1}: {str(e)}")
            success = False

    return success

def store_embedding(embedding, metadata, content_hash):
    vector = np.array(embedding, dtype='float32')
    if vector.size == 0:
        logging.error("Embedding vector is empty. Skipping storage.")
        return

    vector = vector / np.linalg.norm(vector)
    timestamp = metadata.get('timestamp', '1970-01-01 00:00:00')

    conn = psycopg2.connect("host=10.79.85.40 dbname=chatbotdb user=chatbotuser password=changeme")
    cursor = conn.cursor()

    insert_query = """
    INSERT INTO embeddings (embedding_vector, timestamp, error_type, summary, part, content_hash)
    VALUES (%s::float4[], %s, %s, %s, %s, %s)
    """
    try:
        cursor.execute(insert_query, (
            vector.tolist(),
            timestamp,
            metadata.get('error_type', 'N/A'),
            metadata.get('summary', ''),
            metadata.get('part', ''),
            content_hash
        ))
        conn.commit()
        logging.info("Successfully stored embedding in PostgreSQL.")
    except Exception as e:
        logging.error(f"Failed to store embedding in PostgreSQL: {str(e)}")
    finally:
        cursor.close()
        conn.close()

def load_credentials():
    try:
        with open(credentials_file, 'r') as file:
            credentials = json.load(file)
            required_keys = ['username', 'password']
            for key in required_keys:
                if key not in credentials:
                    raise KeyError(f"Missing required credential: {key}")
            logger.info("Credentials loaded successfully.")
            return credentials
    except FileNotFoundError:
        logger.error(f"Credentials file {credentials_file} not found.")
    except KeyError as e:
        logger.error(f"Missing key in credentials: {str(e)}")
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {credentials_file}")
    except Exception as e:
        logger.error(f"Unexpected error loading credentials from {credentials_file}: {str(e)}")
    return None
        
def get_db_connection(host_ip='10.79.85.40'):
    try:
        if not host_ip:
            host_ip = os.getenv('DB_HOST', '10.79.85.40')  # Ensure a fallback host IP
        return psycopg2.connect(
            host=host_ip,
            dbname=os.getenv('DB_NAME', 'chatbotdb'),
            user=os.getenv('DB_USER', 'chatbotuser'),
            password=os.getenv('DB_PASSWORD', 'changeme')
        )
    except Exception as e:
        logger.error(f"Error establishing a database connection: {str(e)}")
        return None

from urllib.parse import urlparse, parse_qs

def extract_space_key_from_url(confluence_url):
    """
    Extract the space key from the given Confluence view page URL.
    """
    try:
        # Parse the URL
        parsed_url = urlparse(confluence_url)
        # Parse the query parameters
        query_params = parse_qs(parsed_url.query)

        # Extract spaceKey
        space_key = query_params.get('spaceKey', [None])[0]
        
        if space_key:
            logger.debug(f"Extracted space key from URL: {space_key}")
            return space_key
        else:
            logger.error("Space key not found in the Confluence URL.")
            return None
    except Exception as e:
        logger.error(f"Error extracting space key from URL: {str(e)}")
        return None


def retry_request(url, headers=None, retries=3, delay=5):
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response
        except RequestException as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(delay)
    logger.error("All attempts to fetch data failed.")
    return None
    
def is_valid_url(url):
    regex = re.compile(
        r'^(?:http|https)://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|'  # ...or ipv4
        r'\[?[A-F0-9]*:[A-F0-9:]+\]?)'  # ...or ipv6
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return re.match(regex, url) is not None
    
# Main function to handle embedding from GitHub and Confluence
def main():
    parser = argparse.ArgumentParser(description="Embed data from various sources.")
    parser.add_argument('--github', type=str, help='URL of the GitHub repository to embed.')
    parser.add_argument('--confluence', type=str, help='Confluence page URL to embed.')
    parser.add_argument('--space_key', type=str, help='Confluence space key.')
    parser.add_argument('--title', type=str, help='Confluence page title.')
    parser.add_argument('--host_ip', type=str, default='10.79.85.40', help='Database host IP.')
    parser.add_argument('--verbose', action='store_true', help="Enable verbose mode for detailed output.")
    parser.add_argument('--org_url', type=str, help='GitHub organization URL to process all repositories.')

    args = parser.parse_args()
    setup_logging(args.verbose)
    list_confluence_spaces("https://confluence.dtc.dish.corp", pat_token)
    
    repos = list_repositories(org_url, git_token)
    
    if not repos:
        logger.error("No repositories found.")
        return

    # Clone each repository
    for repo in repos:
        repo_name = repo.get("name")
        ssh_url = repo.get("ssh_url_to_repo")

        if ssh_url:
            clone_repository(ssh_url, repo_name)
    
    # Log arguments
    logger.info(f"Received arguments: {args}")


    # Embed individual GitHub repository
    if args.github:
        repo_path = clone_or_pull_github_repo(args.github)
        if repo_path:
            content = extract_github_content(repo_path)
            for item in content:
                logger.info(f"Content snippet from {item['file_path']}: {item['content'][:100]}")  # Log first 100 chars
                metadata = {
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'error_type': 'N/A',
                    'summary': f"Content from file: {item['file_path']}",
                    'part': 'GitHub'
                }
                embed_and_store_content(item['content'], metadata, host_ip=args.host_ip)

    # Embed all repositories from a GitHub organization
    if args.org_url:
        org_name = extract_org_name_from_url(args.org_url)
        if org_name:
            process_organization_repos(org_name, git_token)

    # Embed Confluence content
    if args.confluence and args.space_key and args.title:
        credentials = load_credentials()
        if credentials:
            confluence_content = fetch_confluence_page(
                space_key=args.space_key,
                title=args.title,
                confluence_url=args.confluence,
                username=credentials.get('confluence_user'),
                api_token=confluence_token
            )
            if confluence_content:
                metadata = {
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'error_type': 'N/A',
                    'summary': f"Content from Confluence page: {args.title}",
                    'part': 'Confluence'
                }
                embed_and_store_content(confluence_content, metadata, host_ip=args.host_ip)

if __name__ == "__main__":
    main()
