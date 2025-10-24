import feedparser
import google.generativeai as genai
import PyRSS2Gen
import datetime
import pytz # For timezone-aware datetime objects
import os
import requests # For fetching web page content
from bs4 import BeautifulSoup # For parsing HTML
from thefuzz import fuzz # For fuzzy string matching

# --- Configuration ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

model = None 
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        generation_config = {
            "temperature": 0.7,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 2048, 
        }
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash-lite",
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        print("Gemini model initialized successfully.")
    except Exception as e:
        print(f"Error initializing Gemini model: {e}")
        model = None
else:
    print("Warning: GEMINI_API_KEY not found. Summarization will be skipped or limited.")

SOURCE_RSS_FEEDS = {
    "Wired": "https://www.wired.com/feed/category/gear/latest/rss",
    "The Verge": "https://www.theverge.com/rss/index.xml",
    "BBC": "http://feeds.bbci.co.uk/news/technology/rss.xml",
    "Reuters": "https://www.reuters.com/rssFeed/technologyNews",
    "The Gardian": "https://www.theguardian.com/uk/technology/rss",
    "CNN": "http://rss.cnn.com/rss/edition_technology.rss",
    "Google News Tech": "https://news.google.com/rss/search?q=technology&hl=en-US&gl=US&ceid=US:en"
}
OUTPUT_RSS_FILE = "summarized_news.xml"
MAX_ITEMS_PER_SOURCE_FEED = 7 # Fetch a few more items to increase chance of finding duplicates
HOURS_WINDOW = 12 # Widen window slightly for topic clustering
TITLE_SIMILARITY_THRESHOLD = 85 # Adjust this (0-100) for title matching sensitivity

# --- Helper Functions ---
def get_aware_datetime(time_struct):
    if time_struct:
        try:
            dt = datetime.datetime(*time_struct[:6])
            return pytz.utc.localize(dt)
        except Exception as e:
            # print(f"Error converting time_struct to datetime: {time_struct}, Error: {e}")
            return None
    return None

def fetch_full_article_text(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        print(f"  Fetching full article content from: {url}")
        response = requests.get(url, headers=headers, timeout=20) # Increased timeout slightly
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        text_parts = []
        # Prioritize common article tags. This can be expanded.
        main_content_tags = soup.find_all(['article', 'main', 'section[class*="article-content"]', 'div[class*="article-body"]', 'div[class*="story-content"]', 'div[class*="main-content"]'])

        content_element = None
        if main_content_tags:
            # Take the one with the most <p> tags or longest text as a heuristic
            best_candidate = None
            max_p_count = -1
            for tag in main_content_tags:
                p_count = len(tag.find_all('p', recursive=False)) # Non-recursive to avoid double counting from nested <article>
                if p_count > max_p_count:
                    max_p_count = p_count
                    best_candidate = tag
                elif p_count == max_p_count and best_candidate and len(tag.get_text()) > len(best_candidate.get_text()):
                     best_candidate = tag
            content_element = best_candidate

        if not content_element: # Fallback if specific tags aren't found or don't yield much
            content_element = soup # Use the whole soup

        paragraphs = content_element.find_all('p') if content_element else []

        for p in paragraphs:
            text_parts.append(p.get_text(separator=' ', strip=True))

        full_text = "\n\n".join(filter(None, text_parts)) # Filter out empty strings

        if not full_text.strip() or len(full_text.strip()) < 200: # Require some substantial text
            print(f"  Could not extract significant text (found {len(full_text.strip())} chars) from {url}.")
            return None

        print(f"  Successfully extracted ~{len(full_text)} characters from {url}.")
        max_chars_for_summary = 25000 
        if len(full_text) > max_chars_for_summary:
            print(f"  Truncating extracted text from {len(full_text)} to {max_chars_for_summary} characters.")
            full_text = full_text[:max_chars_for_summary]
        return full_text
    except requests.exceptions.Timeout:
        print(f"  Timeout fetching URL {url}")
    except requests.exceptions.RequestException as e:
        print(f"  Error fetching URL {url}: {e}")
    except Exception as e:
        print(f"  Error parsing content from {url}: {e}")
    return None

def summarize_text_with_gemini(text_to_summarize, article_title="this article"):
    if not model: return "Summary not available (Gemini model not initialized)."
    if not text_to_summarize or len(text_to_summarize.strip()) < 100: return "Summary not available (insufficient content)."
    try:
        prompt = (
            f"عيد كتابة المقال الإخباري التالي بالعامية المصرية"
f"​التعليمات:​ التركيز: ركز على الأحداث الرئيسية، الشخصيات المهمة، الأرقام التواريخ، والنتائج النهائية."
f"​التنظيم: قسم التلخيص في فقرات متوسطة، بحد أقصى 4 فقرات. تجنب أي ذكر انك بتلخص المقال ورد بالتلخيص واعادة الكتابة فورا بدون اي اعادة للتعليمات."
f"​الدقة: حافظ على كل المعلومات الجوهرية من المقال الأصلي بدون ما تفقد أي تفاصيل مهمة."
f"​الإيجاز: استخدم لغة بسيطة ومباشرة وجذابة واستخدم الايموجي لو لقيت فيه حاجه لكدة بدون زيادة استخدام عالفاضي، وتجنب التكرار والحشو والكلام اللي مالوش لازمة."
f"​البحث الإضافي: ابحث في مصادر خارجية عشان توضح خلفية الخبر أو تكمل أي معلومة ناقصة، بحيث يكون التلخيص شامل ومفهوم.\n\n:\n{text_to_summarize}"
        )
        
      #  prompt = (
      #      f"Please provide a comprehensive, multi-paragraph summary of the following news article titled '{article_title}'. "
      #      f"Cover the main points, key arguments, and any significant conclusions. "
      #      f"Aim for a summary that captures the essence of the full article, "
      #      f"as if explaining it to someone who hasn't read it.\n\nArticle Content:\n{text_to_summarize}"
      #  )


        
        print(f"  Sending text (first 100 chars: '{text_to_summarize[:100]}...') to Gemini for detailed summarization.")
        response = model.generate_content(prompt)
        if response.candidates and response.candidates[0].content.parts:
            summary_text = response.candidates[0].content.parts[0].text.strip()
            print(f"  Gemini detailed summary received (first 100 chars: '{summary_text[:100]}...').")
            return summary_text
        else:
            block_reason = response.prompt_feedback.block_reason if hasattr(response, 'prompt_feedback') and response.prompt_feedback else "Unknown"
            finish_reason = response.candidates[0].finish_reason if response.candidates else "Unknown"
            print(f"  Gemini API response issue for detailed summary. Block reason: {block_reason}, Finish reason: {finish_reason}")
            return "Detailed summary generation failed (API response structure issue)."
    except Exception as e:
        print(f"  Error during Gemini API call for detailed summary: {e}")
        return f"Detailed summary generation error: {type(e).__name__} - {e}"

def group_articles(articles):
    """Groups articles based on title similarity."""
    story_clusters = []
    processed_indices = set()

    for i, article1 in enumerate(articles):
        if i in processed_indices:
            continue

        current_cluster = [article1]
        processed_indices.add(i)

        for j, article2 in enumerate(articles):
            if j <= i or j in processed_indices: # Don't compare with self or already processed/grouped
                continue

            # Compare titles using fuzzy matching
            # token_sort_ratio is good for titles where word order might change slightly
            similarity_score = fuzz.token_sort_ratio(article1['title'], article2['title'])

            if similarity_score >= TITLE_SIMILARITY_THRESHOLD:
                # Optional: Add a check for pub_date proximity if titles are similar
                # e.g., if abs((article1['pub_date'] - article2['pub_date']).total_seconds()) < SOME_THRESHOLD_IN_SECONDS:
                current_cluster.append(article2)
                processed_indices.add(j)

        story_clusters.append(current_cluster)

    print(f"Formed {len(story_clusters)} story clusters from {len(articles)} articles.")
    return story_clusters

# --- Main Logic ---
def main():
    print("Starting RSS summarization script with duplicate detection & topic ranking...")
    all_candidate_articles = []
    now_utc = datetime.datetime.now(pytz.utc)
    time_cutoff = now_utc - datetime.timedelta(hours=HOURS_WINDOW)

    print(f"Fetching news published after: {time_cutoff.strftime('%Y-%m-%d %H:%M:%S %Z')} from various sources.")

    for feed_name, feed_url in SOURCE_RSS_FEEDS.items():
        print(f"Processing feed: {feed_name} ({feed_url})")
        try:
            parsed_feed = feedparser.parse(feed_url)
            if parsed_feed.bozo: print(f"  Warning: Feed '{feed_name}' may be malformed. Reason: {parsed_feed.bozo_exception}")
        except Exception as e:
            print(f"  Could not parse feed {feed_name}: {e}"); continue

        items_from_this_feed = 0
        for entry in parsed_feed.entries:
            if items_from_this_feed >= MAX_ITEMS_PER_SOURCE_FEED: break

            pub_date_parsed = entry.get("published_parsed")
            pub_date = get_aware_datetime(pub_date_parsed)

            if pub_date and pub_date >= time_cutoff:
                title = entry.get("title", "No Title").strip()
                link = entry.get("link", "#")

                all_candidate_articles.append({
                    "title": title,
                    "link": link,
                    "pub_date": pub_date,
                    "content_for_summary": entry.get("summary", entry.get("description", "")), # Initial content
                    "full_content_fetched": False, # Flag to see if we fetched full text
                    "source_feed": feed_name,
                    "id": entry.get("id", link) # Unique ID for the article
                })
            items_from_this_feed += 1

    print(f"Total articles fetched initially (before filtering & content extraction): {len(all_candidate_articles)}")

    # Sort by pub_date initially to process newest first if needed, though grouping changes order
    all_candidate_articles.sort(key=lambda x: x["pub_date"], reverse=True)

    # Group similar articles (potential duplicates)
    if not all_candidate_articles:
        print("No articles fetched. Exiting.")
        # Create an empty or minimal RSS feed if desired
        # (Code for empty feed from previous versions can be added here)
        return

    story_clusters = group_articles(all_candidate_articles)

    # Rank story clusters: Primary by repetition (size of cluster), secondary by recency of newest article in cluster
    ranked_stories = []
    for cluster in story_clusters:
        if not cluster: continue
        # Get the most recent pub_date from the articles in this cluster
        # And select the article with the most recent pub_date as representative for sorting
        cluster.sort(key=lambda x: x['pub_date'], reverse=True)
        representative_article = cluster[0] # Newest in this cluster

        ranked_stories.append({
            "representative_article_title": representative_article['title'],
            "repetition_count": len(cluster),
            "most_recent_pub_date": representative_article['pub_date'],
            "articles_in_cluster": cluster, # Keep all articles in cluster for potential later use or inspection
            "representative_article_for_summary": representative_article # This is the one we'll try to summarize
        })

    # Sort ranked_stories: more repetitions first, then by most recent date
    ranked_stories.sort(key=lambda x: (x["repetition_count"], x["most_recent_pub_date"]), reverse=True)

    final_rss_items = []
    if not ranked_stories:
        print("No story clusters formed. Creating a default item.")
        # (Code for empty/default feed item)
        rss_item = PyRSS2Gen.RSSItem(
            title="No prominent news topics identified",
            link=f"https://{os.environ.get('GITHUB_REPOSITORY_OWNER', 'your-username')}.github.io/{os.environ.get('GITHUB_REPOSITORY_NAME', 'your-repo-name')}/",
            description=f"Could not identify any news story clusters based on repetition in the last {HOURS_WINDOW} hours.",
            pubDate=datetime.datetime.now(pytz.utc)
        )
        final_rss_items.append(rss_item)
    else:
        top_story_cluster_info = ranked_stories[0]
        article_to_summarize = top_story_cluster_info["representative_article_for_summary"]

        print(f"\nTop story selected for summarization (Repetition: {top_story_cluster_info['repetition_count']}):")
        print(f"  Title: '{article_to_summarize['title']}'")
        print(f"  Source: {article_to_summarize['source_feed']}")
        print(f"  Published: {article_to_summarize['pub_date']}")
        print(f"  Original Link: {article_to_summarize['link']}")
        print(f"  Cluster contains {len(top_story_cluster_info['articles_in_cluster'])} similar articles:")
        for i, art_in_cluster in enumerate(top_story_cluster_info['articles_in_cluster']):
            if i < 3: # Print first 3 for brevity
                 print(f"    - '{art_in_cluster['title']}' from {art_in_cluster['source_feed']}")
            elif i == 3:
                 print(f"    ... and {len(top_story_cluster_info['articles_in_cluster']) - 3} more.")
                 break


        # Fetch full content for the chosen article
        full_content = fetch_full_article_text(article_to_summarize['link'])
        text_for_gemini = full_content if full_content else article_to_summarize['content_for_summary'] # Fallback to RSS content

        ai_summary = "Detailed summary placeholder..."
        if model and GEMINI_API_KEY and text_for_gemini:
             ai_summary = summarize_text_with_gemini(text_for_gemini, article_to_summarize['title'])
        elif not GEMINI_API_KEY: ai_summary = "Detailed summary not available (API key missing)."
        elif not model: ai_summary = "Detailed summary not available (Gemini model initialization failed)."
        else: ai_summary = "Detailed summary not available (No content for summarization)."

        rss_item = PyRSS2Gen.RSSItem(
            title=f"{article_to_summarize['title']}",
            link=article_to_summarize['link'],
            description=(f"{ai_summary}\n\n"
                         f"Source for summary: {article_to_summarize['source_feed']}"),
            guid=PyRSS2Gen.Guid(article_to_summarize['link']), # Use original link for GUID
            pubDate=article_to_summarize['pub_date']
        )
        final_rss_items.append(rss_item)
        print(f"  Final item summary (first 150 chars): {ai_summary[:150]}...")

    project_page_url = f"https://{os.environ.get('GITHUB_REPOSITORY_OWNER', 'your-username')}.github.io/{os.environ.get('GITHUB_REPOSITORY_NAME', 'your-repo-name')}/"
    rss_feed = PyRSS2Gen.RSS2(
        title="My AI Top News Summary (Ranked by Topic Repetition)",
        link=project_page_url,
        description=(f"The most prominent news story (ranked by repetition across sources and recency from last {HOURS_WINDOW}hrs), "
                     f"with detailed multi-paragraph AI summary."),
        lastBuildDate=datetime.datetime.now(pytz.utc),
        items=final_rss_items,
        language="en-us",
    )

    try:
        with open(OUTPUT_RSS_FILE, "w", encoding="utf-8") as f:
            rss_feed.write_xml(f, encoding="utf-8")
        print(f"\nSuccessfully generated RSS feed with ranked and summarized news: {OUTPUT_RSS_FILE}")
    except IOError as e:
        print(f"Error writing RSS file: {e}")

    print("Script for ranked summaries finished.")

if __name__ == "__main__":
    main()
