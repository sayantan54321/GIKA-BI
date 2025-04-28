import json
import requests
from typing import Dict, Any
import logging
from urllib.parse import urlparse
import os
import google.generativeai as genai
import time
import random
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def get_domain(url: str) -> str:
    parsed_url = urlparse(url)
    return parsed_url.netloc if parsed_url.netloc else parsed_url.path.split("/")[0]


def format_url(url, page_url):
    if url and url[0] == "/":
        return f"{get_domain(page_url)}{url}"
    return ""


def infer_ollama_model(prompt: str, user_question: str):
    """
    Sends a request to the OLLama model API and returns the response.

    Args:
        prompt (str): The prompt to send to the model.
        image (str, optional): An image to include with the prompt.

    Returns:
        str: The response from the API.
    """
    logger.info("Generating response with ollama")
    url = "http://localhost:6001/api/chat"
    payload: Dict[str, Any] = {
        # "model": "gemma3:4b-it-fp16",
        # "model": "gemma3:27b-it-q8_0",
        # "model": "gemma3:12b",
        "model": "llama3.3:70b-instruct-q8_0",
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_question},
        ],
        # "messages": [
        #     {"role": "system", "content": prompt, "images": None},
        #     {"role": "user", "content": user_question, "images": None},
        # ],
        "stream": False,
    }

    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            result = response.json()["message"]["content"]
            logger.debug(f"Received response from Ollama API: {result}")
            return result
        else:
            logger.error(f"Ollama API returned status code: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to reach Ollama API: {e}")
        return None


def infer_gemini_model(prompt: str, user_question: str):
    GOOGLE_API_KEY = [
                    "AIzaSyBMzZ1SY0NCIaYV2Fcm8xryiB4EXapidLU",
                    "AIzaSyD7Lo1JUlgC2c9z5O6d9fy1mAAzzgnCrpA",
                    "AIzaSyB2F_bhKZARc3gM6Jeo8yBqvXLam3Rog_k",
                    "AIzaSyAuHXvD8f6Lx6CNawq6qtRuMnT9_pIGHUI",
                    "AIzaSyCAYJnBS8PA7thnrTQOG-3GeEtqTDyjnys",
                    "AIzaSyCKjPJJ5HmzJjBV_BYrJSeasz-z8q9z4lk"
                ]
    API_KEY = random.choice(GOOGLE_API_KEY)
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt + "\n\n" + user_question)
    # time.sleep(5)
    return response.text


def batch_list(input_list, batch_size):
    """Splits input_list into a list of lists of size batch_size."""
    return [
        input_list[i : i + batch_size] for i in range(0, len(input_list), batch_size)
    ]


import ast
import re


def parse_list_from_string(s: str) -> list[str]:
    """Parses a list of strings from a formatted string in the form '*[*]*', ignoring any surrounding nonsense."""
    match = re.search(r"\[.*?\]", s)
    if match:
        try:
            return ast.literal_eval(match.group())
        except (SyntaxError, ValueError):
            pass
    raise ValueError("Invalid input format")


def extract_product_description_urls(llm_response, query_urls):
    truth_list = parse_list_from_string(llm_response)

    result = set()

    for i in range(min(len(query_urls), len(truth_list))):
        if truth_list[i].lower() == "yes":
            result.add(query_urls[i])

    return list(result)


def get_pc_urls(urls_info):
    return [url["url"] for url in urls_info if url["is_product_catalogue_url"] == True]


def get_pd_urls(urls_info):
    return [url["url"] for url in urls_info if url["is_product_catalogue_url"] == False]


def get_query_urls(urls_info):
    links = set()
    for url in urls_info:
        if url["is_product_catalogue_url"]:
            page_url = url["url"]
            for link in url["links"]:
                links.add(format_url(link, page_url))

    return list(links)


def get_promt():
    with open(
        "/home/aadey/workspace/data_enrichment/product-urls-scraper/url_filter_prompt.txt",
        "r",
        encoding="utf-8",
    ) as f:
        return f.read()


def filter_urls(urls_info, query):

    pc_urls = get_pc_urls(urls_info)
    pd_urls = get_pd_urls(urls_info)
    links = get_query_urls(urls_info)
    prompt = get_promt()

    link_batches = batch_list(links, 20)

    result = []

    final_product_urls = set(pd_urls)

    for link_batch in link_batches:
        user_context = {
            "product_catalogue_urls": pc_urls[:2],
            "product_description_urls": pd_urls[:2],
            "query_title": query,
            "query_urls": link_batch,
        }

        llm_response = infer_gemini_model(prompt, str(user_context))

        product_urls = extract_product_description_urls(llm_response, link_batch)

        final_product_urls.update(product_urls)

        result.append(
            {"links": link_batch, "results": product_urls, "llm_response": llm_response}
        )

        with open("./url_filter_result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)

    return list(final_product_urls)
