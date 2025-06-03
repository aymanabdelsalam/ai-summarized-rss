import feedparser
import google.generativeai as genai
import PyRSS2Gen
import datetime
import pytz # For timezone-aware datetime objects
import os

# --- Configuration ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

model = None # Initialize model to None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        generation_config = {
            "temperature": 0.6,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 150,
        }
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash-latest", # Or "gemini-pro" if preferred & within free tier limits
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        print("Gemini model initialized successfully.")
    except Exception as e:
        print(f"Error initializing Gemini model: {e}")
        model = None # Ensure model is None if initialization fails
else:
    print("Warning: GEMINI_API_KEY not found in environment variables. Summarization will be skipped.")

SOURCE_RSS_FEEDS = {
    "BBC World": "http://feeds.bbci.co.uk/news/world/rss.xml",
    "Reuters World": "http://feeds.reuters.com/Reuters/worldNews",
    "NPR News": "https://feeds.npr.org/1001/rss.xml",
    "Google News Tech": "https://news.google.com/rss/search?q=technology&hl=en-US&gl=US&ceid=US:en"
    # Add reliable AP feed if you find one. Some previous examples might be unstable.
}
OUTPUT_RSS_FILE = "summarized_news.xml" # This file will be created in the repo root
MAX_ITEMS_PER_SOURCE_FEED = 5
HOURS_WINDOW = 6

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

def summarize_text_with_gemini(text_to_summarize):
    if not model:
        return "Summary not available (Gemini model not initialized or API key missing)."
    if not text_to_summarize or len(text_to_summarize.strip()) < 30: # Minimum content length for a decent summary
        return "Summary not available (insufficient content from feed)."
    try:
        prompt = f"Summarize the following news article concisely in 1-2 compelling sentences, focusing on the most critical information:\n\n{text_to_summarize}"
        print(f"Sending text to Gemini for summarization (first 100 chars): {text_to_summarize[:100]}...")
        response = model.generate_content(prompt)

        # Detailed logging of response for debugging
        # print(f"Gemini raw response: {response}") # Potentially very verbose

        if response.candidates and response.candidates[0].content.parts:
            summary_text = response.candidates[0].content.parts[0].text.strip()
            print(f"Gemini summary received: {summary_text[:100]}...")
            return summary_text
        else:
            # Log reasons for failure if possible
            block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else "Unknown"
            finish_reason = response.candidates[0].finish_reason if response.candidates else "Unknown"
            print(f"Gemini API response issue. Block reason: {block_reason}, Finish reason: {finish_reason}")
            if response.candidates and not response.candidates[0].content.parts:
                print("No parts in candidate content.")
            return "Summary generation failed (API response structure)."
    except Exception as e:
        print(f"Error during Gemini API call: {e}")
        return f"Summary generation error: {type(e).__name__} - {e}"

# --- Main Logic ---
def main():
    print("Starting RSS summarization script...")
    candidate_articles = []
    now_utc = datetime.datetime.now(pytz.utc)
    time_cutoff = now_utc - datetime.timedelta(hours=HOURS_WINDOW)

    print(f"Fetching news published after: {time_cutoff.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    for feed_name, feed_url in SOURCE_RSS_FEEDS.items():
        print(f"Processing feed: {feed_name} ({feed_url})")
        try:
            parsed_feed = feedparser.parse(feed_url)
            if parsed_feed.bozo: # Check for malformed feed
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
                content_to_summarize = entry.get("summary", entry.get("description"))
                if not content_to_summarize and entry.get('content'):
                    if isinstance(entry.content, list) and len(entry.content) > 0:
                        content_to_summarize = entry.content[0].value

                print(f"  Found recent article: '{title}' ({pub_date.strftime('%Y-%m-%d %H:%M')})")
                candidate_articles.append({
                    "title": title,
                    "link": link,
                    "pub_date": pub_date,
                    "content": content_to_summarize if content_to_summarize else "No distinct summary/content found in feed item.",
                    "source_feed": feed_name
                })
            processed_in_feed_count += 1

    print(f"Total candidate articles found across all feeds: {len(candidate_articles)}")

    final_rss_items = []
    if not candidate_articles:
        print("No recent articles found matching criteria. Creating a default item.")
        rss_item = PyRSS2Gen.RSSItem(
            title="No recent news",
            link=f"https://{os.environ.get('GITHUB_REPOSITORY_OWNER', 'your-username')}.github.io/{os.environ.get('GITHUB_REPOSITORY_NAME', 'your-repo-name')}/",
            description=f"No news items found in the last {HOURS_WINDOW} hours from the top {MAX_ITEMS_PER_SOURCE_FEED} of selected feeds.",
            pubDate=datetime.datetime.now(pytz.utc)
        )
        final_rss_items.append(rss_item)
    else:
        candidate_articles.sort(key=lambda x: x["pub_date"], reverse=True)
        top_article = candidate_articles[0]
        print(f"\nSelected top article: '{top_article['title']}' from {top_article['source_feed']}")

        ai_summary = "Summary placeholder (API call skipped or failed)" # Default
        if model and GEMINI_API_KEY: # Only attempt summary if model is ready
             ai_summary = summarize_text_with_gemini(top_article['content'])
        elif not GEMINI_API_KEY:
            ai_summary = "Summary not available (API key missing)."
        else: # Model initialization failed
            ai_summary = "Summary not available (Gemini model initialization failed)."


        rss_item = PyRSS2Gen.RSSItem(
            title=f"{top_article['title']}", # Keep original title, summary is in description
            link=top_article['link'],
            description=f"{ai_summary}\n\nSource: {top_article['source_feed']}",
            guid=PyRSS2Gen.Guid(top_article['link']),
            pubDate=top_article['pub_date']
        )
        final_rss_items.append(rss_item)
        print(f"  Final item description (summary part): {ai_summary[:150]}...")

    project_page_url = f"https://{os.environ.get('GITHUB_REPOSITORY_OWNER', 'your-username')}.github.io/{os.environ.get('GITHUB_REPOSITORY_NAME', 'your-repo-name')}/"

    rss_feed = PyRSS2Gen.RSS2(
        title="My AI Top News Summary",
        link=project_page_url,
        description=f"The single most recent top news item from selected feeds (last {HOURS_WINDOW}hrs, top {MAX_ITEMS_PER_SOURCE_FEED}), summarized by AI.",
        lastBuildDate=datetime.datetime.now(pytz.utc),
        items=final_rss_items,
        language="en-us",
    )

    try:
        with open(OUTPUT_RSS_FILE, "w", encoding="utf-8") as f:
            rss_feed.write_xml(f, encoding="utf-8")
        print(f"\nSuccessfully generated RSS feed: {OUTPUT_RSS_FILE}")
    except IOError as e:
        print(f"Error writing RSS file: {e}")

    print("Script finished.")

if __name__ == "__main__":
    main()
