from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import asyncio
from crawl4ai import AsyncWebCrawler
from srapped_raw_data_structured_raw_data import *
from filter_raw_scrapped_data_using_gemini import *
import pandas as pd
import re
import json
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = FastAPI()

class ScrapeRequest(BaseModel):
    keywords: List[str]
    filtering_prompt: str

class WebScrapingAgent:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    async def scrape_page(self, url: str):
        print(f"Scraping: {url}")
        async with AsyncWebCrawler(verbose=True) as crawler:
            crawler.browser_config = {
                "browser_type": "chromium",
                "headless": True,
                "viewport": {"width": 1920, "height": 1080},
            }
            result = await crawler.arun(
                url=url,
                crawl_options={
                    "include_tags": ["p", "article", "div", "section", "h1", "h2", "h3"],
                    "exclude_tags": ["script", "style", "nav", "footer", "header", "aside"],
                }
            )
            if result.success:
                raw_text = result.markdown
                chunks = self.text_splitter.split_text(raw_text) if raw_text else []
                return {
                    "url": url,
                    "title": result.metadata.get("title", "Unknown Title") if result.metadata else "Unknown Title",
                    "raw_content": raw_text,
                    "filtered_chunks": chunks,
                    "success": True
                }
            else:
                return {
                    "url": url,
                    "title": "Scraping Failed",
                    "raw_content": "",
                    "filtered_chunks": [],
                    "success": False,
                    "error": result.error_message
                }

    async def scraping_articles(self, all_raw_contents: list, output_csv: str = "pmc_results.csv", output_json: str = "pmc_results.json", parallel_requests: int = 10):
        combined_raw = "\n".join(all_raw_contents)
        pmc_ids = re.findall(r'PMC\d+', combined_raw)
        pmc_ids = list(set(pmc_ids))
        data = []

        article_urls = [f"https://pmc.ncbi.nlm.nih.gov/articles/{pmc}/" for pmc in pmc_ids]

        # Process in batches of 'parallel_requests' (now always 10)
        for i in range(0, len(article_urls), parallel_requests):
            batch_urls = article_urls[i:i+parallel_requests]
            results = await asyncio.gather(*(self.scrape_page(url) for url in batch_urls))
            for result in results:
                pmc = re.search(r'PMC\d+', result["url"]).group(0) if result["success"] else None
                title = result["title"] if result["success"] else "Unknown Title"
                content = result["raw_content"] if result["success"] else None
                data.append({"pmcid": pmc, "url": result["url"], "title": title, "content": content})

        df = pd.DataFrame([{"pmcid": d["pmcid"], "url": d["url"]} for d in data if d["pmcid"]])
        df.to_csv(output_csv, index=False)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


@app.post("/scrape_and_filter")
async def scrape_and_filter(request: ScrapeRequest):
    agent = WebScrapingAgent()
    keywords = request.keywords
    filtering_prompt = request.filtering_prompt
    all_raw_contents = []

    # Build a single search term by joining keywords with '+'
    search_term = "+".join(keywords)
    for page_number in range(1, 2):  # You can change range for more pages
        url = f"https://pmc.ncbi.nlm.nih.gov/search/?term={search_term}&sort=relevance&page={page_number}&ac=no"
        result = await agent.scrape_page(url)
        if result["success"]:
            all_raw_contents.append(result["raw_content"])

    # Always scrape 10 articles in parallel
    await agent.scraping_articles(all_raw_contents, parallel_requests=10)
    await process_urls_from_csv("pmc_results.csv", "scraped_articles.csv")
    result = filter_scraped_data("./vector_db_storage", "scraped_articles.json", filtering_prompt, "pmc_results.csv")
    return result