from datetime import datetime
import pytz
import requests
from duckduckgo_search import DDGS 
from bs4 import BeautifulSoup
import re


def extract_search_query(user_input):
    # Define patterns to match common search phrases
    patterns = [
        r'search for (.+)',
        r'find (.+)',
        r'look up (.+)',
        r'what is (.+)',
        r'what are (.+)',
        r'current time in (.+)',
        r'weather in (.+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, user_input.lower())
        if match:
            return match.group(1).strip()

    # If no pattern matches, return the original input
    return user_input.strip()

def do_web_search(query):
    # Try the first web search method
    query = extract_search_query(query)
    try:
        search_result = first_web_search(query)
        if not search_result or "An error occurred" in search_result:
            raise ValueError("First web search failed")
    except Exception as e:
        print(f"First web search failed: {e}")
        # Try the second web search method
        try:
            search_result = second_web_search(query)
            if not search_result or "An error occurred" in search_result:
                raise ValueError("Second web search failed")
        except Exception as e:
            print(f"Second web search failed: {e}")
            return "An error occurred during the web search."
    return search_result


def get_current_time(timezone='UTC'):
    tz = pytz.timezone(timezone)
    current_time = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
    return f"The current time in {timezone} is {current_time}."
	
def get_weather(location):
    # Geocoding to get latitude and longitude from location name
    geocode_url = f'https://geocoding-api.open-meteo.com/v1/search?name={location}'
    geocode_response = requests.get(geocode_url)
    if geocode_response.status_code == 200:
        geocode_data = geocode_response.json()
        if 'results' in geocode_data and len(geocode_data['results']) > 0:
            latitude = geocode_data['results'][0]['latitude']
            longitude = geocode_data['results'][0]['longitude']
        else:
            return "Unable to find the specified location."
    else:
        return "Unable to retrieve location information."

    # Get weather data
    weather_url = 'https://api.open-meteo.com/v1/forecast'
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'current_weather': True,
        'timezone': 'auto',
        'hourly': 'temperature_2m,relativehumidity_2m'
    }
    weather_response = requests.get(weather_url, params=params)
    if weather_response.status_code == 200:
        weather_data = weather_response.json()
        if 'current_weather' in weather_data:
            temperature = weather_data['current_weather']['temperature']
            windspeed = weather_data['current_weather']['windspeed']
            weather_info = f"The current temperature in {location} is {temperature}°C with a wind speed of {windspeed} m/s."
            return weather_info
        else:
            return "Unable to retrieve weather information."
    else:
        return "Unable to retrieve weather information."
        
def first_web_search(query): 
    try:
        with DDGS() as ddgs:
            # Pass 'max_results' to 'ddgs.text()', not to 'DDGS()'
            results = ddgs.text(
                query,
                region='wt-wt',
                safesearch='Moderate',
                timelimit=None,
                max_results=5  # Correct placement of 'max_results'
            )
            results_list = list(results)
            if results_list:
                # Extract the snippet or body from the first result
                snippet = results_list[0].get('body') or results_list[0].get('snippet') or 'No description available.'
                return snippet
            else:
                return "No results found."
    except Exception as e:
        return f"An error occurred during the web search: {str(e)}"

def second_web_search(query): 
    # Perform a web search by scraping DuckDuckGo's HTML page
    params = {
        'q': query,
    }
    headers = {
        'User-Agent': 'Mozilla/5.0'
    }
    response = requests.get('https://html.duckduckgo.com/html/', params=params, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    results = soup.find_all('a', class_='result__a')
    if results:
        # Get the text of the first result
        snippet = results[0].get_text()
        return snippet
    else:
        return None