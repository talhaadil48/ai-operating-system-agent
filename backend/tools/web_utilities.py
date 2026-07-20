import requests
from langchain_core.tools import tool
from backend.logging_config import get_logger
from backend.llm.factory import get_llm
from langchain_core.messages import SystemMessage

log = get_logger(__name__)

# --- Helper functions ---

def _clean_html(html_content: str) -> str:
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_content, "html.parser")
        # Remove unwanted tags
        for element in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            element.decompose()
        text = soup.get_text(separator="\n")
    except ImportError:
        # Fallback to standard html.parser
        from html.parser import HTMLParser
        class HTMLTextExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self.result = []
                self.ignore = False
            def handle_starttag(self, tag, attrs):
                if tag in ["script", "style", "nav", "footer", "header", "aside"]:
                    self.ignore = True
            def handle_endtag(self, tag):
                if tag in ["script", "style", "nav", "footer", "header", "aside"]:
                    self.ignore = False
            def handle_data(self, data):
                if not self.ignore:
                    self.result.append(data)
        parser = HTMLTextExtractor()
        parser.feed(html_content)
        text = "\n".join(parser.result)

    # Normalize whitespace
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = "\n\n".join(chunk for chunk in chunks if chunk)
    return text

def _chunk_text(text: str, max_chars: int = 5000) -> list[str]:
    chunks = []
    current_chunk = []
    current_len = 0
    for paragraph in text.split("\n\n"):
        if current_len + len(paragraph) > max_chars:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
            current_chunk = [paragraph]
            current_len = len(paragraph)
        else:
            current_chunk.append(paragraph)
            current_len += len(paragraph) + 2
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    return chunks

# --- Tools ---

@tool
def scrape_webpage(url: str) -> str:
    """Extract clean main body text from a webpage URL.

    Capped to prevent exceeding LLM token/rate limits. Use this when you need
    to read the contents of a specific website.

    Args:
        url: The webpage URL to scrape (e.g. 'https://en.wikipedia.org/wiki/Artificial_intelligence').
    """
    log.info("[web_utils] Scraping URL: %s", url)
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        clean_text = _clean_html(response.text)
        
        # Cap the output length at 5,000 characters to protect token limits
        max_output_chars = 5000
        if len(clean_text) > max_output_chars:
            return (
                clean_text[:max_output_chars]
                + f"\n\n... [TRUNCATED - Webpage text was too long ({len(clean_text)} chars). "
                f"Use the 'summarize_webpage' tool to get a full summary without hitting token limits]"
            )
        return clean_text
    except Exception as e:
        log.exception("[web_utils] Failed to scrape webpage")
        return f"Failed to scrape webpage: {e}"

@tool
def summarize_webpage(url: str) -> str:
    """Fetch and summarize the entire contents of a webpage URL.

    Automatically splits large articles into smaller chunks and summarizes them
    to avoid rate limits, returning a concise summary.

    Args:
        url: The webpage URL to summarize.
    """
    log.info("[web_utils] Summarizing URL: %s", url)
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        clean_text = _clean_html(response.text)
        if not clean_text.strip():
            return "Webpage is empty or contains no extractable text."
            
        chunks = _chunk_text(clean_text, max_chars=4000)
        llm = get_llm()
        
        chunk_summaries = []
        for idx, chunk in enumerate(chunks, start=1):
            log.info("[web_utils] Summarizing chunk %d/%d", idx, len(chunks))
            prompt = (
                "Summarize the following section of a webpage. Highlight key arguments, "
                "data, or conclusions in bullet points. Be concise.\n\n"
                f"Webpage Snippet:\n{chunk}"
            )
            resp = llm.invoke([SystemMessage(content=prompt)])
            chunk_summaries.append(resp.content.strip())
            
        # Consolidate if multiple chunks
        if len(chunk_summaries) == 1:
            return chunk_summaries[0]
            
        log.info("[web_utils] Consolidating %d summaries", len(chunk_summaries))
        combined_summaries = "\n\n".join(chunk_summaries)
        consolidation_prompt = (
            "Consolidate the following webpage summaries into a single cohesive, structured summary "
            "with key bullet points. Do not lose any crucial information.\n\n"
            f"Chunk Summaries:\n{combined_summaries}"
        )
        resp = llm.invoke([SystemMessage(content=consolidation_prompt)])
        return resp.content.strip()
        
    except Exception as e:
        log.exception("[web_utils] Failed to summarize webpage")
        return f"Failed to summarize webpage: {e}"

@tool
def check_website_status(url: str) -> str:
    """Check if a website is online and responsive.

    Args:
        url: The URL to check (e.g., 'https://google.com').
    """
    log.info("[web_utils] Checking status for URL: %s", url)
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(url, headers=headers, timeout=10)
        status_code = response.status_code
        reason = response.reason
        elapsed = response.elapsed.total_seconds()
        
        status = "ONLINE" if status_code < 400 else "OFFLINE / ERROR"
        return (
            f"Website: {url}\n"
            f"Status: {status} (HTTP {status_code} {reason})\n"
            f"Response Time: {elapsed:.3f} seconds"
        )
    except requests.exceptions.RequestException as e:
        return f"Website: {url}\nStatus: OFFLINE\nError Detail: {e}"

@tool
def get_weather(location: str) -> str:
    """Get the current weather details for a given city or location.

    Args:
        location: The name of the city, region, or address (e.g., 'London', 'New York', 'Tokyo').
    """
    log.info("[web_utils] Fetching weather for location: %s", location)
    try:
        # Step 1: Geocoding location to latitude/longitude
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={requests.utils.quote(location)}&count=1&format=json"
        geo_resp = requests.get(geo_url, timeout=10)
        geo_resp.raise_for_status()
        geo_data = geo_resp.json()
        
        results = geo_data.get("results")
        if not results:
            return f"Could not find coordinates for location '{location}'. Please try a larger city."
            
        loc_data = results[0]
        lat = loc_data["latitude"]
        lon = loc_data["longitude"]
        name = loc_data.get("name", location)
        country = loc_data.get("country", "")
        
        # Step 2: Fetching weather from Open-Meteo
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        weather_resp = requests.get(weather_url, timeout=10)
        weather_resp.raise_for_status()
        weather_data = weather_resp.json()
        
        curr = weather_data.get("current_weather")
        if not curr:
            return f"Weather data currently unavailable for {name}."
            
        temp = curr.get("temperature")
        windspeed = curr.get("windspeed")
        winddirection = curr.get("winddirection")
        weathercode = curr.get("weathercode")
        
        # Open-Meteo weather codes mapping
        wmo_codes = {
            0: "Clear sky",
            1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Fog", 48: "Depositing rime fog",
            51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
            61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
            71: "Slight snow fall", 73: "Moderate snow fall", 75: "Heavy snow fall",
            77: "Snow grains",
            80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
            85: "Slight snow showers", 86: "Heavy snow showers",
            95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail"
        }
        condition = wmo_codes.get(weathercode, "Unknown")
        
        return (
            f"=== Weather for {name}, {country} ===\n"
            f"Condition:   {condition}\n"
            f"Temperature: {temp}°C\n"
            f"Wind Speed:  {windspeed} km/h (Direction: {winddirection}°)"
        )
    except Exception as e:
        log.exception("[web_utils] Failed to get weather information")
        return f"Failed to retrieve weather for '{location}': {e}"
