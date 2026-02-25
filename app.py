import streamlit as st
from newspaper import Article
from transformers import pipeline
import time

# Page Configuration
st.set_page_config(page_title="5-Article Summary Tool", layout="centered")

# Load the Summarization Model
@st.cache_resource
def load_model():
    return pipeline("summarization", model="facebook/bart-large-cnn")

try:
    summarizer = load_model()
except:
    st.error("Error loading AI model. Please refresh and try again.")
    st.stop()

def get_article_text(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except:
        return None

def summarize_text(text):
    if not text:
        return ""
    chunk = text[:1500]
    if len(chunk) < 50:
        return text
    try:
        summary = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except:
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
                
                if raw_text:
                    summary = summarize_text(raw_text)
                    all_summaries.append(summary)
                    st.success("Done")
                else:
                    st.error(f"Failed to scrape: {url}")

            if all_summaries:
                st.divider()
                st.subheader("🧠 Combined Master Summary")
                
                # Combine all summaries and summarize again
                combined = " ".join(all_summaries)
                final_summary = summarize_text(combined)
                
                st.write(final_summary)
                
                st.download_button("Download Text", final_summary, "summary.txt", "text/plain")