import feedparser
import google.generativeai as genai
import PyRSS2Gen
import datetime
import pytz # For timezone-aware datetime objects
import os
import requests # For fetching web page content
from bs4 import BeautifulSoup # For parsing HTML

# --- Configuration ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

model = None 
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        generation_config = {
            "temperature": 0.7, # Might need adjustment for longer summaries
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 800, # Increased for multi-paragraph summaries (max for gemini-1.5-flash is 8192, pro is much higher but check current free tier limits)
        }
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash-latest", # Flash is good for speed/cost.
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        print("Gemini model initialized successfully.")
    except Exception as e:
        print(f"Error initializing Gemini model: {e}")
        model = None
else:
    print("Warning: GEMINI_API_KEY not found in environment variables. Summarization will be skipped or limited.")

SOURCE_RSS_FEEDS = {
    "Tech Crunch": "https://techcrunch.com/feed",
    "Venture Beat": "https://feeds.feedburner.com/venturebeat/SZYF",
    "The Verge": "https://www.theverge.com/rss/index.xml",
    "Tech Republic": "https://www.techrepublic.com/rssfeeds/articles/?feedType=rssfeeds&sort=latest"
}
OUTPUT_RSS_FILE = "summarized_news.xml"
MAX_ITEMS_PER_SOURCE_FEED = 5 # Process top 5 from each source
HOURS_WINDOW = 2

# --- Helper Functions ---
def get_aware_datetime(time_struct):
    if time_struct:
        try:
            dt = datetime.datetime(*time_struct[:6])
            return pytz.utc.localize(dt)
        except Exception as e:
            print(f"Error converting time_struct to datetime: {time_struct}, Error: {e}")
            return None
    return None

def fetch_full_article_text(url):
    """
    Fetches and extracts text content from an article URL.
    This is a generic extractor and its success varies by website structure.
    """
    try:
        headers = { # Common headers to mimic a browser
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        print(f"Fetching full article content from: {url}")
        response = requests.get(url, headers=headers, timeout=15) # 15-second timeout
        response.raise_for_status() # Raises an HTTPError for bad responses (4XX or 5XX)

        soup = BeautifulSoup(response.content, 'html.parser')

        # Generic text extraction:
        # Try to find common article containers first. This list can be expanded.
        possible_containers = soup.find_all(['article', 'main', 'div[class*="content"]', 'div[class*="article-body"]', 'div[id*="article"]'])

        text_parts = []
        if possible_containers:
            for container in possible_containers:
                paragraphs = container.find_all('p')
                for p in paragraphs:
                    text_parts.append(p.get_text(separator=' ', strip=True))
        else: # Fallback if no specific containers found
            paragraphs = soup.find_all('p')
            for p in paragraphs:
                text_parts.append(p.get_text(separator=' ', strip=True))

        full_text = "\n\n".join(text_parts)

        if not full_text.strip():
            print(f"  Could not extract significant text from {url} using generic paragraph search.")
            return None

        print(f"  Successfully extracted ~{len(full_text)} characters from {url}.")
        # Truncate very long texts to avoid exceeding Gemini's input token limits (adjust as needed)
        # Gemini 1.5 Flash has a large context window, but still good to be mindful.
        # A typical token is ~4 chars. 1,000,000 token limit is huge.
        # Let's cap it at something reasonable like 20000 characters for input.
        max_chars_for_summary = 20000 
        if len(full_text) > max_chars_for_summary:
            print(f"  Truncating extracted text from {len(full_text)} to {max_chars_for_summary} characters for summarization.")
            full_text = full_text[:max_chars_for_summary]
        return full_text

    except requests.exceptions.RequestException as e:
        print(f"  Error fetching URL {url}: {e}")
    except Exception as e:
        print(f"  Error parsing content from {url}: {e}")
    return None


def summarize_text_with_gemini(text_to_summarize, article_title="this article"):
    if not model:
        return "ملخص غير متاح (لم يتم تهيئة نموذج Gemini أو مفتاح API مفقود)." # "Summary not available (Gemini model not initialized or API key missing)."
    if not text_to_summarize or len(text_to_summarize.strip()) < 100:
        return "ملخص غير متاح (محتوى غير كافي لملخص تفصيلي)." # "Summary not available (insufficient content provided for a detailed summary)."
    try:
        # Updated prompt for a longer, multi-paragraph summary IN EGYPTIAN ARABIC SLANG
        prompt = (
            f"لو سمحت، اعمل ملخص شامل من كذا فقرة للمقالة الإخبارية دي بعنوان '{article_title}' باللهجة المصرية العامية (بتاعة الشارع). "
            f"الملخص المفروض يغطي النقط الأساسية، الحجج المهمة، وأي استنتاجات ضرورية. "
            f"الهدف إن الملخص يكون بيفهم اللي مقراش المقالة إيه اللي حصل بالظبط، كأنك بتحكيله الحكاية بالبلدي كدة. "
            f"استخدم مفردات بسيطة وعامية وماتكترش في الكلام الرسمي.\n\nمحتوى المقالة:\n{text_to_summarize}"
        )

        # English translation of the new prompt for clarity:
        # "Please provide a comprehensive, multi-paragraph summary of this news article titled '{article_title}' in Egyptian colloquial Arabic (street slang). "
        # "The summary should cover the main points, important arguments, and any necessary conclusions. "
        # "The goal is for the summary to make someone who hasn't read the article understand exactly what happened, as if you're telling them the story in a very casual, everyday way. "
        # "Use simple, colloquial vocabulary and don't use too much formal language.\n\nArticle Content:\n{text_to_summarize}"

        print(f"Sending text (first 100 chars: '{text_to_summarize[:100]}...') to Gemini for detailed Egyptian Arabic slang summarization.")
        response = model.generate_content(prompt)

        if response.candidates and response.candidates[0].content.parts:
            summary_text = response.candidates[0].content.parts[0].text.strip()
            print(f"Gemini detailed Egyptian Arabic summary received (first 100 chars: '{summary_text[:100]}...').")
            return summary_text
        else:
            block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else "غير معروف" # "Unknown"
            finish_reason = response.candidates[0].finish_reason if response.candidates else "غير معروف" # "Unknown"
            error_message_arabic = "مشكلة في استجابة Gemini API لملخص اللهجة المصرية. سبب الحجب: {block_reason}, سبب الإنهاء: {finish_reason}"
            print(error_message_arabic.format(block_reason=block_reason, finish_reason=finish_reason))
            return "فشل إنشاء ملخص تفصيلي باللهجة (مشكلة في هيكل استجابة API)." # "Detailed slang summary generation failed (API response structure issue)."
    except Exception as e:
        error_message_arabic = f"حدث خطأ أثناء استدعاء Gemini API لملخص اللهجة المصرية: {type(e).__name__} - {e}"
        print(error_message_arabic)
        return error_message_arabic # Return the Arabic error message


# --- Main Logic ---
def main():
    print("Starting RSS summarization script (for detailed summaries)...")
    candidate_articles = []
    now_utc = datetime.datetime.now(pytz.utc)
    time_cutoff = now_utc - datetime.timedelta(hours=HOURS_WINDOW)

    print(f"Fetching news published after: {time_cutoff.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    for feed_name, feed_url in SOURCE_RSS_FEEDS.items():
        print(f"Processing feed: {feed_name} ({feed_url})")
        try:
            parsed_feed = feedparser.parse(feed_url)
            if parsed_feed.bozo:
                print(f"  Warning: Feed '{feed_name}' may be malformed. Bozo reason: {parsed_feed.bozo_exception}")
        except Exception as e:
            print(f"  Could not parse feed {feed_name}: {e}")
            continue

        processed_in_feed_count = 0
        for entry in parsed_feed.entries:
            if processed_in_feed_count >= MAX_ITEMS_PER_SOURCE_FEED:
                break

            title = entry.get("title", "No Title")
            link = entry.get("link", "#")
            pub_date_parsed = entry.get("published_parsed")
            pub_date = get_aware_datetime(pub_date_parsed)

            if pub_date and pub_date >= time_cutoff:
                print(f"  Found recent RSS item: '{title}' ({pub_date.strftime('%Y-%m-%d %H:%M')}) from {link}")

                # Attempt to fetch full article text
                full_content = fetch_full_article_text(link)

                text_for_summary = full_content
                if not text_for_summary:
                    # Fallback to RSS summary if full content fetching fails
                    print(f"    Full content fetch failed for '{title}'. Falling back to RSS summary/description.")
                    text_for_summary = entry.get("summary", entry.get("description"))
                    if not text_for_summary and entry.get('content'):
                        if isinstance(entry.content, list) and len(entry.content) > 0:
                            text_for_summary = entry.content[0].value

                if text_for_summary:
                    candidate_articles.append({
                        "title": title,
                        "link": link,
                        "pub_date": pub_date,
                        "content_for_summary": text_for_summary, # This is now potentially the full article text
                        "source_feed": feed_name
                    })
                else:
                    print(f"    Skipping '{title}' due to lack of content for summarization.")

            processed_in_feed_count += 1

    print(f"Total candidate articles with content for summarization: {len(candidate_articles)}")

    final_rss_items = []
    if not candidate_articles:
        print("No recent articles found with sufficient content. Creating a default item.")
        # (Same default item creation as before)
        rss_item = PyRSS2Gen.RSSItem(
            title="No recent news with processable content",
            link=f"https://{os.environ.get('GITHUB_REPOSITORY_OWNER', 'your-username')}.github.io/{os.environ.get('GITHUB_REPOSITORY_NAME', 'your-repo-name')}/",
            description=f"No news items found in the last {HOURS_WINDOW} hours from which full content could be reliably extracted for detailed summary.",
            pubDate=datetime.datetime.now(pytz.utc)
        )
        final_rss_items.append(rss_item)
    else:
        candidate_articles.sort(key=lambda x: x["pub_date"], reverse=True)
        top_article = candidate_articles[0] # We still pick only one "top" article
        print(f"\nSelected top article for detailed summary: '{top_article['title']}' from {top_article['source_feed']}")

        ai_summary = "Detailed summary placeholder (API call skipped, model issue, or content insufficient)"
        if model and GEMINI_API_KEY:
             ai_summary = summarize_text_with_gemini(top_article['content_for_summary'], top_article['title'])
        elif not GEMINI_API_KEY:
            ai_summary = "Detailed summary not available (API key missing)."
        else:
            ai_summary = "Detailed summary not available (Gemini model initialization failed)."

        rss_item = PyRSS2Gen.RSSItem(
            title=f"{top_article['title']}",
            link=top_article['link'],
            description=f"\n{ai_summary}\n\n المصدر: {top_article['source_feed']}", # Newlines for better readability of multi-para summary
            guid=PyRSS2Gen.Guid(top_article['link']),
            pubDate=top_article['pub_date']
        )
        final_rss_items.append(rss_item)
        print(f"  Final item description (summary part - first 150 chars): {ai_summary[:150]}...")

    # (RSS feed generation part remains largely the same as before)
    project_page_url = f"https://{os.environ.get('GITHUB_REPOSITORY_OWNER', 'your-username')}.github.io/{os.environ.get('GITHUB_REPOSITORY_NAME', 'your-repo-name')}/"
    rss_feed = PyRSS2Gen.RSS2(
        title="My AI Detailed News Summary",
        link=project_page_url,
        description=f"The single most recent top news item (full article summarized) from selected feeds (last {HOURS_WINDOW}hrs, top {MAX_ITEMS_PER_SOURCE_FEED}), with detailed multi-paragraph AI summary.",
        lastBuildDate=datetime.datetime.now(pytz.utc),
        items=final_rss_items,
        language="en-us",
    )

    try:
        with open(OUTPUT_RSS_FILE, "w", encoding="utf-8") as f:
            rss_feed.write_xml(f, encoding="utf-8")
        print(f"\nSuccessfully generated RSS feed with detailed summary: {OUTPUT_RSS_FILE}")
    except IOError as e:
        print(f"Error writing RSS file: {e}")

    print("Script for detailed summaries finished.")

if __name__ == "__main__":
    main()
