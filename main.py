import os
import re
import time
import math
import json
import logging
import base64
from threading import Thread, Lock as ThreadLock
from io import BytesIO
from PIL import Image
from bs4 import BeautifulSoup
from typing import Dict, Any, List, Tuple
from pathlib import Path
import requests
from flask import Flask, request, jsonify
import hashlib
from constants import prompt
from url_filter import filter_urls
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from crawl4ai import AsyncWebCrawler
from transformers import pipeline
from functools import lru_cache, wraps
from tqdm import tqdm

logger = logging.getLogger("main")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

app = Flask(__name__)

# Create a session for connection pooling
http_session = requests.Session()
http_session.mount('http://', requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20))
http_session.mount('https://', requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20))

# Result cache
result_cache = {}

def sanitize_query(query):
    return re.sub(r"[^\w\s-]", "", query).strip().replace(" ", "_")

# Function to cache results
def cache_result(func):
    cache = {}
    @wraps(func)
    def wrapper(*args, **kwargs):
        key = str(args) + str(sorted(kwargs.items()))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    return wrapper

def search(query, num_results=10, max_retries=3):
    """Search function with retry mechanism and fallback"""
    for attempt in range(max_retries):
        try:
            logger.info(f"Searching for '{query}', attempt {attempt+1}/{max_retries}")
            payload = {"query": query, "num_results": num_results}
            
            # Use a longer timeout
            response = http_session.post(
                "http://localhost:1243/search", 
                json=payload, 
                timeout=120  # 2 minutes timeout
            )
            
            response.raise_for_status()
            search_results = response.json().get("result", [])            
            valid_results = [result["url"] for result in search_results if result and isinstance(result, dict) and "url" in result]
            logger.info(f"Search successful for '{query}', got {len(valid_results)} results")
            return valid_results
        
        except requests.exceptions.Timeout:
            logger.error(f"Search timeout for '{query}' (attempt {attempt+1}/{max_retries})")
            if attempt == max_retries - 1:
                # On final attempt, return hardcoded fallback URLs
                return get_fallback_urls_for_query(query)
            time.sleep(5 * (attempt + 1))  # Exponential backoff
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Search error for '{query}': {e} (attempt {attempt+1}/{max_retries})")
            if attempt == max_retries - 1:
                # On final attempt, return hardcoded fallback URLs
                return get_fallback_urls_for_query(query)
            time.sleep(5 * (attempt + 1))  # Exponential backoff
            
    # If we get here, all attempts failed
    return []

def get_fallback_urls_for_query(query):
    """Return hardcoded fallback URLs based on query"""
    query_lower = query.lower()
    
    fallback_urls = []
    
    # Add site-specific URLs based on query
    if "flipkart" in query_lower:
        fallback_urls.append("https://www.flipkart.com/grocery-supermart-store")
    
    if "bigbasket" in query_lower:
        fallback_urls.append("https://www.bigbasket.com")

    if "amazon" in query_lower:
        fallback_urls.append("https://www.amazon.in")
        
    if "blinkit" in query_lower:
        fallback_urls.append("https://blinkit.com")
        
    if "zepto" in query_lower:
        fallback_urls.append("https://www.zeptonow.com")
        
    if "swiggy" in query_lower:
        fallback_urls.append("https://www.swiggy.com/stores/instamart")
        
    if "zomato" in query_lower:
        fallback_urls.append("https://www.zomato.com")
    
    if "fruit" in query_lower or "fruits" in query_lower:
        if "flipkart" in query_lower:
            fallback_urls.append("https://www.flipkart.com/grocery/fresh-fruits-vegetables/fruits/pr?sid=73z,1ux,9mg")
        if "bigbasket" in query_lower:
            fallback_urls.append("https://www.bigbasket.com/pc/fruits-vegetables/fresh-fruits/")
    
    # Add general fallbacks if we have none so far
    if not fallback_urls:
        fallback_urls = [
            "https://www.flipkart.com",
            "https://www.bigbasket.com"
        ]
    
    logger.info(f"Using fallback URLs for query '{query}': {fallback_urls}")
    return fallback_urls

def infer_ollama_model(prompt: str, user_question: str, img: str):
    logger.info("Generating response with gpt")
    from openai import OpenAI

    client = OpenAI()
    try:
        if not img.startswith("data:image/"):
            img = f"data:image/jpeg;base64,{img}"

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.4,
            max_tokens=500,
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": img,
                            },
                        }
                    ],
                },
            ],
        )

        result = completion.choices[0].message.content
        return result
    except Exception as e:
        logger.error(f"Error sending image to API: {e}")
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.4,
            max_tokens=500,
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": f"Unable to process image. Please analyze based on URL: {user_question}",
                },
            ],
        )

        result = completion.choices[0].message.content
        return result

@cache_result
def generate_osint_query(user_query):
    query_prompt = f"""
You are a highly skilled assistant specialized in transforming OSINT queries into concise Google search queries.
Your task is to process the following natural language query and output a search string that:
1. Extracts the primary subject or keyword (for example, "chips").
3. For a single website, outputs: "<keyword> site:<domain>".
4. For multiple websites, output in a list: ```
<keyword> site:<domain1>
<keyword> site:<domain2>
```
.
5. Ensure that the final output is succinct and directly usable in a Google search.
    
Examples:
- Query: "give me overall analysis of chips available in flipkart.com"
  Output (a list of queries): 
  ```
  chips site:flipkart.com
  ```

- Query: "give me overall analysis of chips available in flipkart.com grocery and bigbasket.com"
  Output (a list of queries): 
  ```
  chips site:flipkart.com AND marketplace:GROCERY
  chips site:bigbasket.com
  ```

**Remember for this websites this should come under the site: operator:**

flipkart grocery-> site:flipkart.com AND marketplace:GROCERY
bigbasket-> site:bigbasket.com
amazon-> site:amazon.in
blinkit-> site:blinkit.com
zepto-> site:zeptonow.com
site:swiggy.com/stores/instamart/
zomato-> site:zomato.com

Now, process the following query:
"""
    from openai import OpenAI

    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.4,
        max_tokens=2048,
        messages=[
            {"role": "system", "content": query_prompt},
            {"role": "user", "content": user_query},
        ],
    )

    result = (
        completion.choices[0].message.content.replace("```", "").strip().split("\n")
    )
    return result
   
def add_url_to_crawler(url, timeout=15):
    payload = {"url": url}
    try:
        response = requests.post("http://localhost:1234/crawl", json=payload, timeout=timeout)
        response.raise_for_status()
        logger.info(f"Started crawl for: {url}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to start crawl for {url}: {e}")
        return None

def fetch_crawled_content(url, timeout=15):
    payload = {"url": url, "format": "html", "screenshot": True}
    try:
        response = requests.post("http://localhost:1234/crawl_status", json=payload, timeout=timeout)
        response.raise_for_status()
        result = response.json()
        crawl_status = result["status"]
        logger.info(f"Crawl Status of url[{url}]: {crawl_status}")
        if result and "content" in result:
            logger.info(f"Crawl completed for {url}")
            return [
                crawl_status,
                {
                    "screenshot": result.get("screenshot", None),
                    "html": result.get("content", None),
                },
            ]
        else:
            logger.info(f"Crawl still in progress for {url}")
            return [crawl_status, None]
    except Exception as e:
        logger.error(f"Error fetching crawled content for {url}: {e}")
        return ["FAILED", None]

def fetch_product_content(url, timeout=15):
    payload = {"url": url, "format": "text"}
    try:
        response = requests.post("http://localhost:1234/crawl_status", json=payload, timeout=timeout)
        response.raise_for_status()
        result = response.json()
        crawl_status = result["status"]
        logger.info(f"Crawl Status of url[{url}]: {crawl_status}")
        if result and "content" in result:
            logger.info(f"Crawl completed for {url}")
            return [
                crawl_status,
                {
                    "screenshot":None,
                    "html": result.get("content", None),
                },
            ]
        else:
            logger.info(f"Crawl still in progress for {url}")
            return [crawl_status, None]
    except Exception as e:
        logger.error(f"Error fetching product content for {url}: {e}")
        return ["FAILED", None]

def save_screenshot(screenshot, output_file="example_screenshot.png"):
    image_data = base64.b64decode(screenshot)
    with open(output_file, "wb") as file:
        file.write(image_data)

def crop_image(base64_string: str) -> str:
    try:
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data))
        width, height = image.size
        left, upper, right, lower = 0, 0, width, min(768, height)
        cropped_image = image.crop((left, upper, right, lower))
        buffered = BytesIO()
        save_format = 'JPEG'
        cropped_image.save(buffered, format=save_format)
        
        cropped_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return cropped_base64
    except Exception as e:
        logger.error(f"Error cropping image: {e}")
        return base64_string

def get_url_webpage(url, max_retries=3, retry_delay=5):
    """Optimized version with shorter retry delay and fewer retries"""
    add_url_to_crawler(url)
    [crawl_status, result] = fetch_crawled_content(url)
    retries = max_retries
    while (not result) and (crawl_status != "FAILED") and (retries > 0):
        time.sleep(retry_delay)  # Reduced from 25s to 5s
        [crawl_status, result] = fetch_crawled_content(url)
        retries -= 1
    return result

def get_url_productpage(url, max_retries=3, retry_delay=5):
    """Optimized version with shorter retry delay and fewer retries"""
    add_url_to_crawler(url)
    [crawl_status, result] = fetch_product_content(url)
    retries = max_retries
    while (not result) and (crawl_status != "FAILED") and (retries > 0):
        time.sleep(retry_delay)  # Reduced from 25s to 5s
        [crawl_status, result] = fetch_product_content(url)
        retries -= 1
    return result

def html_to_soup(html: str) -> BeautifulSoup:
    """Use lxml parser for better performance"""
    try:
        soup = BeautifulSoup(html, "lxml")
        return soup
    except Exception as e:
        logger.error(f"Error parsing HTML: {e}")
        return BeautifulSoup("", "lxml")  # Return empty soup on error

def get_all_links(soup: BeautifulSoup) -> list:
    links = []
    try:
        for a_tag in soup.find_all("a", href=True):
            links.append(a_tag["href"])
        return links
    except Exception as e:
        logger.error(f"Error getting links: {e}")
        return links

def get_all_classnames(soup: BeautifulSoup) -> list:
    classnames = set()
    try:
        for element in soup.find_all(class_=True):
            classes = element.get("class", [])
            for cls in classes:
                classnames.add(cls)
        return list(classnames)
    except Exception as e:
        logger.error(f"Error getting classnames: {e}")
        return list(classnames)

def create_folder(original_query):
    base_folder = "./search_results"
    os.makedirs(base_folder, exist_ok=True)
    query_folder = os.path.join(base_folder, sanitize_query(original_query))
    os.makedirs(query_folder, exist_ok=True)
    return query_folder

def save_html_file(product_url, original_query, html_content):
    logger.info(
        f"[IMPORTANT] Entered save_html_file - args({product_url}, {original_query})"
    )
    query_folder = create_folder(original_query)
    hashed_file_name = hashlib.md5(product_url.encode("utf-8")).hexdigest() + ".txt"
    file_path = os.path.join(query_folder, hashed_file_name)

    logger.info(f"[IMPORTANT] Path for saving html file: {file_path}")

    with open(file_path, "w", encoding="utf-8") as html_file:
        html_file.write(html_content)

    logger.info(f"Saved HTML for URL: {product_url} at {file_path}")

# def process_single_url(product_url, original_query):
#     """Process a single URL with appropriate error handling"""
#     if not product_url:
#         return
    
#     # Normalize URL
#     normalized_url = product_url
#     if product_url.startswith("www"):
#         normalized_url = "https://" + product_url
        
#     logger.info(f"Processing URL: {normalized_url}")
    
#     try:
#         # Send just one URL at a time
#         payload = {"urls": [normalized_url], "screenshot":False}
#         response = http_session.post(
#             "http://localhost:1113/crawl", 
#             json=payload, 
#             timeout=60  # Increased timeout
#         )
#         response.raise_for_status()
        
#         crawled_content = response.json()
#         if normalized_url in crawled_content and "html" in crawled_content[normalized_url]:
#             save_html_file(
#                 product_url, 
#                 original_query, 
#                 crawled_content[normalized_url]["html"]
#             )
#             logger.info(f"Successfully processed URL: {normalized_url}")
#         else:
#             logger.warning(f"No HTML content for URL: {normalized_url}")
            
#     except Exception as e:
#         logger.error(f"Error processing URL {normalized_url}: {e}")

# def process_urls_sequentially(product_urls, original_query):
#     """Process URLs one at a time to not overwhelm the crawler"""
#     logger.info(f"Processing {len(product_urls)} URLs sequentially for query: {original_query}")
    
#     for url in product_urls:
#         if not url:
#             continue
            
#         normalized_url = url
#         if url.startswith("www"):
#             normalized_url = "https://" + url
            
#         logger.info(f"Processing URL: {normalized_url}")
        
#         try:
#             # Send just one URL at a time
#             payload = {"urls": [normalized_url], "screenshot":False}
#             response = http_session.post(
#                 "http://localhost:1112/crawl",
#                 json=payload, 
#                 timeout=60  # Increased timeout
#             )
#             response.raise_for_status()
            
#             crawled_content = response.json()
#             if normalized_url in crawled_content and crawled_content[normalized_url] and "html" in crawled_content[normalized_url]:
#                 save_html_file(
#                     normalized_url, 
#                     original_query, 
#                     crawled_content[normalized_url]["html"]
#                 )
#                 logger.info(f"Successfully processed URL: {normalized_url}")
#             else:
#                 logger.warning(f"No HTML content for URL: {normalized_url}")
                
#             # Add a significant delay between requests to allow crawler to recover
#             time.sleep(5)
            
#         except Exception as e:
#             logger.error(f"Failed to process URL {normalized_url}: {e}")
#             # Continue with next URL
#             time.sleep(3)  # Short delay before trying next URL

def process_urls_sequentially(product_urls, original_query, batch_size=10):
    """Process URLs in batches of specified size to not overwhelm the crawler"""
    logger.info(f"Processing {len(product_urls)} URLs in batches of {batch_size} for query: {original_query}")    
    valid_urls = []
    for url in product_urls:
        if not url:
            continue
            
        normalized_url = url
        if url.startswith("www"):
            normalized_url = "https://" + url
        
        valid_urls.append(normalized_url)
    
    for i in range(0, len(valid_urls), batch_size):
        batch = valid_urls[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1} with {len(batch)} URLs")
        
        try:
            payload = {"urls": batch, "screenshot": False}
            response = http_session.post(
                "http://localhost:1112/crawl",
                json=payload, 
                timeout=120
            )
            response.raise_for_status()
            
            crawled_content = response.json()
            
            for url in batch:
                if url in crawled_content and crawled_content[url] and "html" in crawled_content[url]:
                    save_html_file(
                        url, 
                        original_query, 
                        crawled_content[url]["html"]
                    )
                    logger.info(f"Successfully processed URL: {url}")
                else:
                    logger.warning(f"No HTML content for URL: {url}")            
            if i + batch_size < len(valid_urls):
                logger.info("Waiting between batches...")
                time.sleep(2)
                
        except Exception as e:
            logger.error(f"Failed to process batch starting at index {i}: {e}")
            time.sleep(5)

def is_product_catalogue(url, query):
    """Check if URL is a product catalog with better error handling"""
    cache_key = f"{url}_{query}"
    if cache_key in result_cache:
        return result_cache[cache_key]
    
    try:
        normalized_url = url
        if url.startswith("www"):
            normalized_url = f"https://{url}"
            
        payload = {"urls": [normalized_url], "screenshot": True}
        
        # Use a longer timeout
        response = http_session.post(
            "http://localhost:1113/crawl", 
            json=payload, 
            timeout=60
        )
        response.raise_for_status()
        crawled_content = response.json()
        
        if not crawled_content:
            logger.warning(f"No content returned for URL: {url}")
            return None

        if normalized_url not in crawled_content:
            logger.warning(f"URL {url} not found in crawled content")
            return None
        
        # Check if error is in the response
        if "error" in crawled_content[normalized_url]:
            logger.warning(f"Error in crawler response for {url}: {crawled_content[normalized_url]['error']}")
            return None
        
        html_content = crawled_content[normalized_url].get("html")
        if html_content is None:
            logger.warning(f"No HTML content for URL: {url}")
            return None

        soup = html_to_soup(html_content)
        links = get_all_links(soup)
        classnames = get_all_classnames(soup)
        screenshot = crawled_content[normalized_url].get("screenshot")
        
        if not screenshot:
            logger.warning(f"No screenshot for URL: {url}")
            result = {
                "url": url,
                "is_product_catalogue_url": False,
                "links": links,
                "classnames": classnames,
            }
            result_cache[cache_key] = result
            return result
        
        # Use the screenshot for LLM analysis
        llm_response = infer_ollama_model(prompt, query, screenshot)
        logger.info(f"URL:{url} - LLM RESPONSE: {llm_response}")
        
        result = {
            "url": url,
            "is_product_catalogue_url": "Yes" in llm_response or "yes" in llm_response,
            "links": links,
            "classnames": classnames,
        }
        
        # Cache the result
        result_cache[cache_key] = result
        return result
    
    except Exception as e:
        logger.error(f"Error checking if URL is product catalogue: {url}, {e}")
        return None

def thread_worker(query, urls, result, _result_lock=None):
    """Process URLs sequentially rather than in parallel to avoid overwhelming crawler"""
    if not urls:
        return
        
    for url in urls:
        try:
            attributed_url = is_product_catalogue(url, query)
            if attributed_url:
                if _result_lock:
                    with _result_lock:
                        result.append(attributed_url)
                else:
                    result.append(attributed_url)
            # Add delay between requests
            time.sleep(3)
        except Exception as e:
            logger.error(f"Error in thread_worker processing URL {url}: {e}")

def get_urls(query):
    search_query = query
    urls = search(search_query, 10)
    return urls
def infer_openai(payload: Dict[str, Any]) -> str:
    """Make inference request to Openai service"""
    response = requests.post("http://localhost:1121/infer_openai", json=payload)
    response.raise_for_status()
    return response.json()["response"]
@cache_result
def infer_gpt(data):
    try:
        import json
        from bs4 import BeautifulSoup        
        if isinstance(data, str):
            text_data = BeautifulSoup(data, "lxml").get_text(separator=' ', strip=True)
        else:
            text_data = data
        if not text_data or len(text_data) < 10:
            logger.warning("Data too short or empty, skipping extraction")
            return "{}"
        response = http_session.post("http://localhost:5006/extract", json={
            "type": "text-generation",
            "batch_messages": [
                {"role": "system", "content": """Extract important `key:value` pairs from this content that can be used to describe the product. You must give your response in correct json format `key:value` without any extra symbol(like \n, \t)  where key is the feature name you are taking and value is the corresponding value. Make sure your response is very brief and not missing any important points. Always include price,discount,rating,quantity as a key if it there in the text. Restrict your response in 10 lines. Also make sure you at the top mention the website name from where the context is given to you.Make sure you don't add any comments only give your final response in the json as your output without mentioning anything about your thinking/reasoning steps.
                **From the given context try to find the vendor/website name from which the product is coming and put that name in the key `website`**.*** Keep the name of  the product always under the key `Product`.*** 
                ***Correct json format example:***
                    {
                        "website": "BigBasket",
                        "Product": "Jackfruit Seeds",
                        "Brand": "fresho!",
                        "Weight": "200 gm",
                        "Price": "₹44.76",
                        "Price_per_gm": "₹0.22",
                        "Country_of_origin": "India"
                    }
                 ***Make sure for each product the json format is strictly consistent and correct and your response for each product should come in correct,parsable json format like above example only.***
                """},
                {"role": "user", "content": data}
            ],
            "temperature": 0.1,
            "max_new_tokens": 1024
        }, timeout=120)
        if response.status_code != 200:
            logger.error(f"API returned non-200 status: {response.status_code}")
            return "{}"
        response.raise_for_status()
        response_json = response.json()
        if isinstance(response_json, dict):
            if "result" in response_json:
                result_str = response_json["result"]                
                try:
                    json_match = re.search(r'\{.*\}', result_str, re.DOTALL)
                    if json_match:
                        extracted_json = json_match.group(0)
                        parsed_json = json.loads(extracted_json)
                        return json.dumps(parsed_json)
                    else:
                        return result_str
                except json.JSONDecodeError:
                    return result_str
            else:
                return json.dumps(response_json)
        else:
            return str(response_json)
    except Exception as e:
        logger.error(f"Error in infer_gpt: {e}")
        return "{}"
        # try:
        #     from openai import OpenAI
        #     client = OpenAI()
        #     data_truncated = data[:10000]  
        #     completion = client.chat.completions.create(
        #         model="gpt-4o-mini",
        #         temperature=0.4,
        #         max_tokens=500,
        #         messages=[
        #             {
        #                 "role": "system",
        #                 "content": """Extract product information from this content in JSON format. Include fields like: website, Product, Price, Quantity, Discount, Rating. Keep the response brief (under 500 characters) and ensure valid JSON format."""
        #             },
        #             {"role": "user", "content": data_truncated},
        #         ],
        #     )
        #     fallback_result = completion.choices[0].message.content
        #     # Try to ensure it's valid JSON by wrapping in quotes if needed
        #     if not (fallback_result.startswith('{') and fallback_result.endswith('}')):
        #         fallback_result = '{"Error": "Could not extract valid product data", "RawExtraction": "' + fallback_result.replace('"', '\\"') + '"}'
        #     return fallback_result
        # except Exception as fallback_error:
        #     logger.error(f"Fallback extraction also failed: {fallback_error}")
        #     return "{}"  # Return empty JSON as last resort

def process_html_file(file_path):
    """Process a single HTML file and extract product info"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = f.read()
        result = infer_gpt(data)
        return result
    except Exception as e:
        logger.error(f"Error processing HTML file {file_path}: {e}")
        return None

@cache_result
def infer_gpt_model(prompt: str, user_question: str):
    try:
        logger.info("Generating response with gpt")
        from openai import OpenAI

        client = OpenAI()
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=2000,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_question},
            ],
        )

        result = completion.choices[0].message.content
        return result
    except Exception as e:
        logger.error(f"Error in infer_gpt_model: {e}")
        return "Failed to generate response due to an error."

# def get_all_crawled_content(original_query):
#     """
#     Retrieve all crawled content from all JSON files in the results directory.
#     Uses sequential processing for stability.
#     """
#     results_path = Path(create_folder(original_query))
#     html_files = list(results_path.glob("*.txt"))
#     logger.info(f"Found {len(html_files)} result files")
    
#     # Process files sequentially
#     content_results = []
#     for file_path in html_files:
#         try:
#             result = process_html_file(file_path)
#             if result:
#                 content_results.append(result)
#         except Exception as e:
#             logger.error(f"Error processing file {file_path}: {e}")
    
#     # Combine results
#     content = "\n" + "-" * 100 + "\n".join(content_results)
    
#     # Save combined content
#     result_path = Path(create_folder(original_query))
#     with open(f"{result_path}/context.txt", "w", encoding="utf-8") as f:
#         f.write(content)
    
#     return content
def get_all_crawled_content(original_query):
    """
    Retrieve all crawled content and process in smaller batches for inference.
    Uses a more resilient approach to batch processing.
    """
    import json
    from bs4 import BeautifulSoup
    
    results_path = Path(create_folder(original_query))
    html_files = list(results_path.glob("*.txt"))
    logger.info(f"Found {len(html_files)} result files")
    
    if not html_files:
        logger.warning("No HTML files found for processing")
        result_path = Path(create_folder(original_query))
        with open(f"{result_path}/context.txt", "w", encoding="utf-8") as f:
            f.write("No product data found for this query.")
        return "No product data found for this query."
    
    file_contents = []
    for file_path in html_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            text_content = BeautifulSoup(content, "lxml").get_text()
            if text_content and len(text_content) > 10:
                file_contents.append({
                    "path": str(file_path),
                    "content": text_content[:5000] 
                })
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
    
    # batch_size = 3
    all_results = []
    
    system_prompt = """Extract important `key:value` pairs from this content that can be used to describe the product. You must give your response in correct json format `key:value` without any extra symbol(like \n, \t)  where key is the feature name you are taking and value is the corresponding value. Make sure your response is very brief and not missing any important points. Always include price,discount,rating,quantity as a key if it there in the text. Restrict your response in 10 lines. Also make sure you at the top mention the website name from where the context is given to you.Make sure you don't add any comments only give your final response in the json as your output without mentioning anything about your thinking/reasoning steps.
    **From the given context try to find the vendor/website name from which the product is coming and put that name in the key `website`**.*** Keep the name of  the product always under the key `Product`.*** 
    
    Example output:
    {
        "website": "BigBasket",
        "Product": "Jackfruit Seeds",
        "Brand": "fresho!",
        "Weight": "200 gm",
        "Price": "₹44.76",
        "Price_per_gm": "₹0.22",
        "Country_of_origin": "India"
    }
    """

    with ThreadPoolExecutor() as executor:
        futures = []
        for item in file_contents:
            payload =  {
                "payload": {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": item["content"]}
                    ]
                },
                "model_name": "gpt-4o-mini"
            }
            futures.append(executor.submit(infer_openai, payload,))
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batches"):
            try:
                result = future.result()
                if result:
                    if isinstance(result, dict):
                        if "result" in result:
                            try:
                                json_match = re.search(r'(\{.*\}|\[.*\])', result["result"], re.DOTALL)
                                if json_match:
                                    json_text = json_match.group(0)
                                    json_obj = json.loads(json_text)
                                    all_results.append(json.dumps(json_obj))
                                else:
                                    all_results.append(result["result"])
                            except json.JSONDecodeError:
                                all_results.append(result["result"])
                        else:
                            all_results.append(json.dumps(result))
                    elif isinstance(result, str):
                        try:
                            json_match = re.search(r'(\{.*\}|\[.*\])', result, re.DOTALL)
                            if json_match:
                                json_obj = json.loads(json_match.group(0))
                                all_results.append(json.dumps(json_obj))
                            else:
                                all_results.append(result)
                        except:
                            all_results.append(result)
                    else:
                        all_results.append(str(result))
            except Exception as e:
                logger.error(f"Error processing batch item: {e}")
    
    # for i in range(0, len(file_contents), batch_size):
    #     batch = file_contents[i:i+batch_size]
    #     logger.info(f"Processing batch {i//batch_size + 1}/{(len(file_contents) + batch_size - 1)//batch_size}")

    #     if not batch:
    #         continue
        # batch_messages=[]
        # for item in batch:
        #     # messages = [
        #     #         {"role": "system", "content": system_prompt},
        #     #         {"role": "user", "content": item["content"]}
        #     #     ]
        #     # batch_messages.append(messages)
        #     try:
        #         messages = [
        #             {"role": "system", "content": system_prompt},
        #             {"role": "user", "content": item["content"]}
        #         ]
        #         response = http_session.post("http://localhost:5006/extract", json={
        #             "batch_messages": [messages],
        #             "type": "text-generation",
        #             "temperature": 0.1,
        #             "max_new_tokens": 2048
        #         }, timeout=60)
                
        #         if response.status_code == 200:
        #             result = response.json()
                    
        #             if isinstance(result, dict):
        #                 if "result" in result:
        #                     try:
        #                         json_match = re.search(r'(\{.*\}|\[.*\])', result["result"], re.DOTALL)
        #                         if json_match:
        #                             json_text = json_match.group(0)
        #                             json_obj = json.loads(json_text)
        #                             all_results.append(json.dumps(json_obj))
        #                         else:
        #                             all_results.append(result["result"])
        #                     except json.JSONDecodeError:
        #                         all_results.append(result["result"])
        #                 else:
        #                     all_results.append(json.dumps(result))
        #             elif isinstance(result, str):
        #                 try:
        #                     json_match = re.search(r'(\{.*\}|\[.*\])', result, re.DOTALL)
        #                     if json_match:
        #                         json_obj = json.loads(json_match.group(0))
        #                         all_results.append(json.dumps(json_obj))
        #                     else:
        #                         all_results.append(result)
        #                 except:
        #                     all_results.append(result)
        #             else:
        #                 all_results.append(str(result))
                    
        #             time.sleep(0.5)
        #         else:
        #             logger.error(f"Request failed with status {response.status_code}: {response.text}")
                    
        #     except Exception as e:
        #         logger.error(f"Error processing item: {str(e)}")
        
        # time.sleep(5)

    valid_results = []
    for result in all_results:
        if not result or result == "{}" or result == "[]":
            continue            
        try:
            if isinstance(result, str):
                if (result.startswith("{") and result.endswith("}")) or (result.startswith("[") and result.endswith("]")):
                    try:
                        parsed = json.loads(result)
                        valid_results.append(result)
                    except json.JSONDecodeError:
                        if result.startswith("[") and "'" in result:
                            import ast
                            try:
                                python_obj = ast.literal_eval(result)
                                valid_results.append(json.dumps(python_obj))
                            except:
                                pass
                else:
                    json_match = re.search(r'(\{.*\}|\[.*\])', result, re.DOTALL)
                    if json_match:
                        extracted = json_match.group(0)
                        try:
                            parsed = json.loads(extracted)
                            valid_results.append(json.dumps(parsed))
                        except json.JSONDecodeError:
                            if "'" in extracted:
                                import ast
                                try:
                                    python_obj = ast.literal_eval(extracted)
                                    valid_results.append(json.dumps(python_obj))
                                except:
                                    pass
            else:
                valid_results.append(json.dumps(result))
        except Exception as e:
            logger.warning(f"Skipping invalid JSON: {result[:100]}... Error: {str(e)}")
    
    if valid_results:
        content = "\n".join(valid_results)
    else:
        content = "No valid product data could be extracted."
    
    result_path = Path(create_folder(original_query))
    with open(f"{result_path}/context.txt", "w", encoding="utf-8") as f:
        f.write(content)
    
    return content

def prepare_context(content):
    return content

def generate_prompt(user_question, context):
    prompt = f"""
                You are a helpful AI assisant that answers questions based on the provided context and based on the given user_question:
                {user_question}
                Don't make up information. Make sure you consider all the products given in the context while giving your answer.
                Process your answer according to user query considering all the products. suppose a price related question is asked by the user then make sure you extract price of all the products given in the context along with some related fields like for price related question related fields are product name, discount, quantity, original price, selling price etc. Also if a query asks about competitive analysis of products between multiple websites/vendors make sure your response includes sufficient number of products from all the vendors/websites asked in the query to give a good competitive analysis.
                Also at the end analysing the entire context give a high level overview of the given context in compliance with the user question.
                CONTEXT:
                {context}
            """
    return prompt

def answer_question(user_question):
    """
    End-to-end process to answer a question based on all search results.
    """
    content_results = get_all_crawled_content(user_question)

    if not content_results:
        return "No crawled content found. Please ensure the search and crawl completed successfully."

    context = prepare_context(content_results)
    prompt = generate_prompt(user_question, context)
    answer = infer_gpt_model(prompt, user_question)

    try:
        from openai import OpenAI

        client = OpenAI()
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.4,
            messages=[
                {
                    "role": "system",
                    "content": f"""
                                    I have a dataset of products in JSON format that includes fields like product name, price, original price, discount, ratings, category, reviews and more and this is the user question:
                                    {user_question}
                                    I want to create a production-ready BI dashboard for competitive analysis using modern JavaScript libraries such as Chart.js and D3.js. Please generate a complete HTML file that:
                                    Parse the data properly. Make sure details of all the products in the given data are also shown in the visualization. While doing competitive analysis, make sure you are not missing any important point or any product.
                                    If in the user question it is mentioned to compare the products of multiple vendors/websites then make sure you include all the products from all the vendors/websites in the given data, also in the charts/graphs you must use some color coding to differentiate between the vendors/websites.
                                    Displays the data in multiple interactive charts/graphs that compare key metrics (like price, discount, and ratings) based on the user question. Also make sure you provide a in depth competitive analysis insights of the products in compliance with the user question.
                                    Uses responsive design with a professional BI look (including proper layout, styling, and modular code).
                                    Incorporates inline comments explaining the code.
                                    The output should include the full HTML, CSS (can be inline or via a CDN), and JavaScript code to create the dashboard, ensuring that it is production-ready. Don't give any explaination or comments. just give the html code don't give ```html or similar in the beginning.
                                """,
                },
                {"role": "user", "content": content_results},
            ],
        )

        html = completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Error generating HTML: {e}")
        html = "<html><body><h1>Error generating visualization</h1></body></html>"

    if not answer:
        return "Failed to get a response."

    return answer, html

generated_html_path = None

@app.route("/process", methods=["POST"])
def process_query():
    global generated_html_path
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Please provide a 'query' in JSON payload."}), 400

    original_query = data["query"]
    sanitized_query = sanitize_query(original_query)
    query_folder = os.path.join("./search_results", sanitized_query)
    query_folder_path = Path(query_folder)
    
    # Check cache first
    if os.path.exists(query_folder):
        logger.info(f"Query folder exists: {query_folder}. Checking cached files.")
        if os.path.exists(f"{query_folder}/{sanitized_query}.html") and os.path.exists(f"{query_folder}/context.txt"):
            
            # Return cached results
            with open(f"{query_folder_path}/context.txt", "r", encoding="utf-8") as f:
                context = f.read()
                if (len(context)>50):
                    prompt = generate_prompt(original_query, context)
                    answer = infer_gpt_model(prompt, original_query)
                    generated_html_path = f"{query_folder_path}/{sanitized_query}.html"
                    return jsonify({"answer": answer, "html_path": generated_html_path})
                else:
                    answer, html = answer_question(original_query)
                    generated_html_path = f"{query_folder_path}/{sanitized_query}.html"
                    
                    with open(generated_html_path, "w", encoding="utf-8") as f:
                        f.write(html)
                    
                    logger.info(f"Answer generated: {answer}")
                    return jsonify({"answer": answer, "html_path": generated_html_path})
    
    # Get OSINT queries
    try:
        queries = generate_osint_query(original_query)
        urls = []
        
        # Get URLs sequentially to avoid overwhelming the search service
        for query in queries:
            logger.info(f"Getting URLs for query: {query}")
            try:
                # Limit to fewer URLs per query
                results = search(query, 10)
                if results:
                    urls.extend(results)
                    logger.info(f"Found {len(results)} URLs for query: {query}")
                time.sleep(2)  # Add delay between search queries
            except Exception as e:
                logger.error(f"Error getting URLs for query '{query}': {e}")
                # Continue with next query
        
        # If we don't have any URLs, try a simple search
        if not urls:
            logger.warning("No URLs from OSINT queries. Trying direct search.")
            try:
                direct_results = search(original_query, 5)
                if direct_results:
                    urls.extend(direct_results)
            except Exception as e:
                logger.error(f"Error in direct search: {e}")
        
        # Filter duplicate and invalid URLs
        unique_urls = []
        seen = set()
        for url in urls:
            if url and url not in seen:
                unique_urls.append(url)
                seen.add(url)
        
        urls = unique_urls
        
        # Save URL list
        with open("./urls.json", "w", encoding="utf-8") as f:
            json.dump(urls, f, indent=4)
        logger.info(f"Retrieved URLs: {urls}")
        
        # Process URLs one by one to check if they are product catalogs
        attributed_urls = []
        for url in urls:
            try:
                result = is_product_catalogue(url, original_query)
                if result:
                    attributed_urls.append(result)
                    logger.info(f"URL {url} is product catalog: {result.get('is_product_catalogue_url', False)}")
                # Add delay between checks
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error checking if URL is product catalog: {url}, {e}")
        
        # Save attributed URLs
        with open("./result.json", "w", encoding="utf-8") as f:
            json.dump(attributed_urls, f, indent=4)
            
        # Filter URLs to get product URLs
        product_urls = filter_urls(attributed_urls, original_query) if attributed_urls else []
        with open("./product_urls.json", "w", encoding="utf-8") as f:
            json.dump(product_urls, f, indent=4)
        
        # Process URLs sequentially
        if product_urls:
            # Choose a smaller subset if there are too many URLs
            if len(product_urls) > 200:
                logger.info(f"Limiting from {len(product_urls)} to 200 product URLs to process")
                product_urls = product_urls[:200]
                
            process_urls_sequentially(product_urls, original_query)
        else:
            logger.warning("No product URLs to process. Creating empty context file.")
            result_path = Path(create_folder(original_query))
            with open(f"{result_path}/context.txt", "w", encoding="utf-8") as f:
                f.write("No product data found for this query.")
        
        # Generate final answer
        try:
            answer, html = answer_question(original_query)
            generated_html_path = f"{query_folder_path}/{sanitized_query}.html"
            
            with open(generated_html_path, "w", encoding="utf-8") as f:
                f.write(html)
            
            logger.info(f"Answer generated: {answer}")
            return jsonify({"answer": answer, "html_path": generated_html_path})
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return jsonify({
                "answer": f"I encountered an issue analyzing data for '{original_query}'. Please try a more specific query or try again later.",
                "error": str(e)
            }), 500
            
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        # Create basic response for error case
        error_message = f"I encountered an issue analyzing data for '{original_query}'. The system may be busy or unable to access the requested information. Please try again later or with a more specific query."
        return jsonify({"answer": error_message, "error": str(e)}), 500

@app.route("/html", methods=["GET"])
def get_html():
    global generated_html_path
    # generated_html_path="/home/aadey/workspace/data_enrichment/product-urls-scraper/search_results/give_me_price_comparison_of_fruits_available_in_flipkart_and_bigbasket/give_me_price_comparison_of_fruits_available_in_flipkart_and_bigbasket.html"
    logger.info(f"Generated HTML path: {generated_html_path}")
    if not generated_html_path or not os.path.exists(generated_html_path):
        return jsonify({"error": "No HTML file has been generated yet."}), 404

    with open(generated_html_path, "r", encoding="utf-8") as f:
        return f.read()

@app.route("/build_graph", methods=["POST"])
def build_graph():
    try:
        data = request.get_json()
        if not data or "query" not in data:
            return jsonify({"error": "Please provide a 'query' in JSON payload."}), 400
        original_query = data["query"]
        sanitized_query = sanitize_query(original_query)
        query_folder = os.path.join("./search_results", sanitized_query)
        context_file_path = os.path.join(query_folder, "context.txt")
        
        if not os.path.exists(context_file_path):
            return jsonify({"error": f"Context file not found at {context_file_path}"}), 404
        NEO4J_URI = "neo4j://91.203.135.146:7687"
        NEO4J_USERNAME = "neo4j"
        NEO4J_PASSWORD = "gikaAdmin1@"
        
        try:
            from langchain.graphs import Neo4jGraph
            kg = Neo4jGraph(
                url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD
            )
            logger.info("Clearing existing graph data...")
            kg.query("MATCH (n) DETACH DELETE n")
            logger.info("Existing graph data cleared successfully")

        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            return jsonify({"error": f"Neo4j connection failed: {str(e)}"}), 500
        
        try:
            from langchain_core.documents import Document
            from langchain_community.graphs.graph_document import GraphDocument, Relationship, Node
            import json
            import re            
            data_list = []
            with open(context_file_path, "r", encoding="utf-8") as f:
                content = f.read()            
            content = re.sub(r'-{5,}', '', content)            
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue                
                json_match = re.search(r'(\{.*\})', line)
                if json_match:
                    try:
                        json_str = json_match.group(0)
                        data_item = json.loads(json_str)
                        if isinstance(data_item, dict) and ("Product" in data_item or "product" in data_item):
                            data_list.append(data_item)
                            logger.info(f"Successfully parsed JSON: {json_str[:50]}...")
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse JSON: {line[:50]}... Error: {e}")            
            logger.info(f"Successfully parsed {len(data_list)} JSON objects")            
            extra_data_path = os.path.join(query_folder, "extra_data.json")
            with open(extra_data_path, "w", encoding="utf-8") as f:
                json.dump(data_list, f, indent=2)            
            graph_documents = []
            for data_item in data_list:
                try:
                    product_name = data_item.get("Product") or data_item.get("product")
                    website = data_item.get("website") or data_item.get("Website") or "Unknown"
                    
                    if not product_name:
                        logger.warning(f"No product name found in: {data_item}")
                        continue                    
                    import hashlib
                    product_id = hashlib.md5(f"{website}-{product_name}".encode()).hexdigest()                    
                    product_node = Node(
                        id=product_id, 
                        type="product", 
                        properties={
                            "name": product_name, 
                            "website": website,
                            "label": product_name
                        }
                    )                    
                    nodes = {product_id: product_node}
                    relationships = []                    
                    for key, value in data_item.items():
                        if key.lower() in ["product", "website"] or not value:
                            continue                        
                        string_value = value
                        if isinstance(value, dict):
                            string_value = json.dumps(value)
                        elif not isinstance(value, str):
                            string_value = str(value)
                        attr_id = hashlib.md5(f"{product_id}-{key}".encode()).hexdigest()
                        attr_node = Node(
                            id=attr_id, 
                            type=key.lower(), 
                            properties={
                                "name": key,
                                "value": string_value,
                                "label": string_value
                            }
                        )
                        nodes[attr_id] = attr_node                        
                        rel = Relationship(
                            source=product_node,
                            target=attr_node,
                            type=key.lower()
                        )
                        relationships.append(rel)                    
                    source_doc = Document(page_content=json.dumps(data_item))                    
                    graph_doc = GraphDocument(
                        nodes=list(nodes.values()),
                        relationships=relationships,
                        source=source_doc
                    )
                    
                    graph_documents.append(graph_doc)
                    
                except Exception as e:
                    logger.error(f"Error creating graph document: {str(e)}")
            
            logger.info(f"Created {len(graph_documents)} graph documents")            
            if graph_documents:
                for i, doc in enumerate(graph_documents):
                    try:
                        kg.add_graph_documents([doc])
                        logger.info(f"Added document {i+1}/{len(graph_documents)}")
                    except Exception as e:
                        logger.error(f"Error adding document {i+1}: {e}")
                
                try:
                    kg.query('MATCH (n) WHERE n.label IS NOT NULL SET n.displayName = n.label')
                    logger.info("Successfully set display properties")
                except Exception as e:
                    logger.warning(f"Error setting display properties: {e}")
                
                return jsonify({
                    "success": True,
                    "message": f"Graph built successfully with {len(graph_documents)} nodes from {len(data_list)} data entries",
                    "graph_url": "http://91.203.135.146:7474/browser/",
                    "cypher_tips": [
                        "MATCH (p:product) RETURN p LIMIT 25;",
                        "MATCH (p:product)-[r]->(a) RETURN p, r, a LIMIT 100;",
                        "MATCH (p:product)-[:price]->(price) RETURN p.name, p.website, price.value ORDER BY p.name, price.value;"
                    ]
                })
            else:
                return jsonify({"error": "No valid product data could be processed into graph documents"}), 400
                
        except Exception as e:
            logger.error(f"Error building graph: {e}")
            return jsonify({"error": f"Graph building failed: {str(e)}"}), 500
            
    except Exception as e:
        logger.error(f"Unexpected error in build_graph endpoint: {e}")
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.route("/offline_search", methods=["POST"])
def offline_search():
    import random
    import google.generativeai as genai
    
    try:
        GOOGLE_API_KEY = [""]
        API_KEY = random.choice(GOOGLE_API_KEY)
        genai.configure(api_key=API_KEY)
        
        data = request.get_json()
        if not data or "original_query" not in data:
            return jsonify({"error": "Please provide an 'original_query' in JSON payload."}), 400
        
        original_query = data["original_query"]
        current_question = data.get("question", "")
        sanitized_query = sanitize_query(original_query)
        query_folder = os.path.join("./search_results", sanitized_query)
        query_folder_path = Path(query_folder)
        
        history_file = os.path.join(query_folder, "chat_history.json")
        
        if not os.path.exists(query_folder):
            return jsonify({"error": "No data found for this query. Please run a full search first."}), 404
        
        # Check for context file
        context_file_path = os.path.join(query_folder, "extra_data.json")
        if not os.path.exists(context_file_path):
            # Try to load from context.txt and parse it
            context_txt_path = os.path.join(query_folder, "context.txt")
            if not os.path.exists(context_txt_path):
                return jsonify({"error": "Context files not found. Please run a full search first."}), 404
                
            # Parse context.txt into JSON
            try:
                with open(context_txt_path, "r") as f:
                    raw_text = f.read()
                entries = re.split(r'-{5,}', raw_text)
                
                data_list = []
                for entry in entries:
                    try:
                        entry = entry.strip()
                        if not entry:
                            continue
                            
                        # Try multiple parsing methods
                        try:
                            outer_dict = eval(entry)
                            if isinstance(outer_dict, dict) and 'result' in outer_dict:
                                result_str = outer_dict['result']
                                result_str = result_str.replace('\\n', '').replace('\\', '')
                                data = json.loads(result_str)
                                data_list.append(data)
                            else:
                                data = outer_dict
                                data_list.append(data)
                        except:
                            try:
                                data = json.loads(entry)
                                data_list.append(data)
                            except:
                                json_match = re.search(r'\{.*\}', entry, re.DOTALL)
                                if json_match:
                                    try:
                                        data = json.loads(json_match.group(0))
                                        data_list.append(data)
                                    except:
                                        continue
                    except:
                        continue
                        
                # Save parsed data
                with open(context_file_path, "w", encoding="utf-8") as f:
                    json.dump(data_list, f, indent=4)
            except Exception as e:
                logger.error(f"Error parsing context.txt: {e}")
                return jsonify({"error": "Failed to parse context data."}), 500
        
        # Load the context data
        with open(context_file_path, "r") as f:
            context_data = json.load(f)
        context = json.dumps(context_data, indent=2)
        
        # Load chat history if it exists
        chat_history = []
        if os.path.exists(history_file):
            try:
                with open(history_file, "r") as f:
                    chat_history = json.load(f)
            except json.JSONDecodeError:
                chat_history = []
        
        # If no question, just return the history and HTML path
        if not current_question:
            return jsonify({
                "chat_history": chat_history,
                "html_path": f"{query_folder}/{sanitized_query}.html" if os.path.exists(f"{query_folder}/{sanitized_query}.html") else None
            })
        
        # Process the current question
        prompt = f"""
        You are a data analyst specialized in e-commerce product data analysis. 
        Analyze the provided JSON data containing product information from various websites.

        Query: {current_question}
        
        Instructions:
        1. Provide a DIRECT and CONCISE analysis answering the query. No introduction or explanation of your process.
        2. Create 2-3 visualizations using html,d3.js,chart.js code that best illustrate your findings and give the clean html code after your textual analysis response.
        3. Each visualization should have a brief title and description of key insights only.
        4. DO NOT repeat the JSON data in your response.
        5. DO NOT explain your reasoning process, just provide findings and visualizations.

        Context:
        {context}
        
        Format your response as:
        1. Direct answer to the query (1-2 paragraphs maximum)
        2. Key findings as bullet points
        3. HTML code for visualizations properly illustrating your findings.
        """
        
        # Use cached response if the same question was asked before
        cached_response = None
        for entry in chat_history:
            if entry.get("question") == current_question:
                cached_response = entry
                break
                
        if cached_response:
            return jsonify({
                "answer": cached_response["answer"],
                "html_path": cached_response["html_path"],
                "chat_history": chat_history
            })
        
        # Generate new response
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            response_text = response.text
        except Exception as e:
            logger.error(f"Error generating response with Gemini: {e}")
            # Fallback to GPT
            response_text = infer_gpt_model(prompt, current_question)
        
        # Clean up the response
        if "```json" in response_text:
            response_text = re.sub(r'```json.*?```', '', response_text, flags=re.DOTALL)
            
        html_pattern = r'<!DOCTYPE html>|<html>'
        html_match = re.search(html_pattern, response_text)
        
        if html_match:
            split_index = html_match.start()
            text_analysis = response_text[:split_index].strip()
            html_code = response_text[split_index:].strip()
            
            # Clean up the text and HTML
            text_analysis = re.sub(r'```html', '', text_analysis)
            text_analysis = re.sub(r'<HTML_START>', '', text_analysis)
            text_analysis = re.sub(r'\n\s*\n', '\n\n', text_analysis)        
            html_code = re.sub(r'```', '', html_code)
            html_code = re.sub(r'</HTML_START>', '', html_code)
            
            # Save the HTML file
            html_path = f"{query_folder}/{sanitized_query}_{len(chat_history)}.html"
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_code)

            # Update chat history
            chat_history.append({
                "question": current_question,
                "answer": text_analysis,
                "html_path": html_path
            })        
            with open(history_file, "w") as f:
                json.dump(chat_history, f, indent=2)
            
            return jsonify({
                "answer": text_analysis,
                "html_path": html_path,
                "chat_history": chat_history
            })
        else:
            # No HTML found, just save the text response
            chat_history.append({
                "question": current_question,
                "answer": response_text,
                "html_path": None
            })        
            with open(history_file, "w") as f:
                json.dump(chat_history, f, indent=2)
            
            return jsonify({
                "answer": response_text,
                "chat_history": chat_history,
                "html_path": None
            })
    except Exception as e:
        logger.error(f"Unexpected error in offline_search: {e}")
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5010, threaded=True)