import asyncio
import pandas as pd
import time
from playwright.async_api import async_playwright

async def scrape_pubmed_article(url: str):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/114.0.0.0 Safari/537.36"
            ),
            locale="en-US",
            bypass_csp=True,
            extra_http_headers={
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "DNT": "1",
                "Upgrade-Insecure-Requests": "1",
            }
        )
        page = await context.new_page()
        await page.goto(url)

        # Extract title
        title = await page.title()

        # Extract article text content by selecting main sections
        selectors = [
            "article",
        ]

        texts = []
        for sel in selectors:
            elements = await page.query_selector_all(sel)
            for el in elements:
                text = await el.inner_text()
                if text:
                    texts.append(text.strip())

        await browser.close()

        full_text = "\n\n".join(texts)
        # Extract PMCID from URL
        pmcid = url.split('/')[-1].replace('/', '')
        
        return {"pmcid": pmcid, "url": url, "title": title, "content": full_text}

async def process_urls_from_csv(input_csv, output_csv):
    # Read URLs from CSV
    df = pd.read_csv(input_csv)
    
    # Check which column contains URLs
    url_column = None
    if 'url' in df.columns:
        url_column = 'url'
    elif 'pmc_url' in df.columns:
        url_column = 'pmc_url'
    else:
        raise ValueError("CSV must contain a 'url' or 'pmc_url' column")
    
    urls = df[url_column].tolist()
    print(f"Found {len(urls)} URLs to scrape")
    
    results = []
    
    # Process each URL
    for i, url in enumerate(urls):
        try:
            print(f"[{i+1}/{len(urls)}] Scraping: {url}")
            result = await scrape_pubmed_article(url)
            results.append(result)
            print(f"✅ Successfully scraped: {url}")
            
            # Add a small delay to avoid rate limiting
            time.sleep(1)
        except Exception as e:
            print(f"❌ Error scraping {url}: {str(e)}")
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"✅ Saved {len(results)} articles to {output_csv}")
    # Save results to JSON
    import json
    output_json = output_csv.replace('.csv', '.json')
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✅ Also saved {len(results)} articles to {output_json}")