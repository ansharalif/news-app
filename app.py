import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import pipeline

# Page Configuration
st.set_page_config(page_title="5-Article Summary Tool", layout="centered")

# Load the Summarization Model
@st.cache_resource
def load_model():
    return pipeline("summarization", model="facebook/bart-large-cnn")

try:
    summarizer = load_model()
except Exception as e:
    st.error(f"Error loading AI model: {e}")
    st.stop()

def get_article_text(url):
    """Scrapes text from URL using BeautifulSoup"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Get text from paragraph tags
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])
        
        return text.strip()
    except Exception as e:
        return None

def summarize_text(text):
    """Summarizes text using AI"""
    if not text:
        return ""
    chunk = text[:1500]
    if len(chunk) < 50:
        return text
    try:
        summary = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return text

# --- UI Layout ---
st.title("📰 Multi-Source News Summarizer")
st.markdown("Enter up to **5 URLs** (one per line) to get a combined summary.")

urls_input = st.text_area("Paste URLs here:", height=150, placeholder="https://bbc.com/news...\nhttps://cnn.com/...")

if st.button("Generate Combined Summary"):
    if not urls_input.strip():
        st.warning("Please enter at least one URL.")
    else:
        url_list = [line.strip() for line in urls_input.split('\n') if line.strip()]
        
        if len(url_list) > 5:
            st.error("Maximum 5 URLs allowed.")
        else:
            all_summaries = []
            
            for i, url in enumerate(url_list):
                st.info(f"Processing {i+1}/{len(url_list)}: {url[:50]}...")
                raw_text = get_article_text(url)
                
                if raw_text and len(raw_text) > 50:
                    summary = summarize_text(raw_text)
                    all_summaries.append(summary)
                    st.success("Done")
                else:
                    st.warning(f"Could not extract text from: {url}")

            if all_summaries:
                st.divider()
                st.subheader("🧠 Combined Master Summary")
                
                combined = " ".join(all_summaries)
                final_summary = summarize_text(combined)
                
                st.write(final_summary)
                
                st.download_button("Download Text", final_summary, "summary.txt", "text/plain")
            else:
                st.error("No valid text could be extracted from the URLs.")
