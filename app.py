import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Page Configuration
st.set_page_config(page_title="5-Article Summary Tool", layout="centered")

# Load T5 Model
@st.cache_resource
def load_model():
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return model, tokenizer

try:
    model, tokenizer = load_model()
except Exception as e:
    st.error(f"Error loading AI model: {e}")
    st.stop()

def count_words(text):
    """Count words in text"""
    return len(text.split())

def get_article_text(url):
    """Scrapes text from URL using BeautifulSoup"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove unwanted elements
        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.decompose()
        
        # Get text from paragraph tags
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])
        
        return text.strip()
    except Exception as e:
        return None

def summarize_text(text, is_final=False):
    """Summarizes text using T5 with word count control"""
    if not text:
        return ""
    
    # Clean up text
    text = ' '.join(text.split())
    
    # Truncate input based on whether it's final or individual
    if is_final:
        max_input = 2000  # Allow more text for final summary
    else:
        max_input = 1500
    
    chunk = text[:max_input]
    
    if len(chunk) < 50:
        return text
    
    try:
        # T5 requires "summarize:" prefix
        input_text = "summarize: " + chunk
        
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        
        # Adjust parameters for longer output
        outputs = model.generate(
            inputs["input_ids"],
            max_length=512,  # Allow longer output
            min_length=100,  # Ensure minimum length
            length_penalty=0.8,  # Encourage longer output
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
        
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Ensure minimum word count by adding context if needed
        word_count = count_words(summary)
        
        if is_final:
            # For final summary, ensure minimum 200 words
            while word_count < 200 and len(chunk) < len(text):
                # Add more content to summary
                additional_text = text[len(chunk):len(chunk)+500]
                if additional_text:
                    chunk += " " + additional_text
                    input_text = "summarize: " + chunk
                    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
                    outputs = model.generate(
                        inputs["input_ids"],
                        max_length=512,
                        min_length=100,
                        length_penalty=0.8,
                        num_beams=4,
                        early_stopping=True,
                        no_repeat_ngram_size=2
                    )
                    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    word_count = count_words(summary)
                else:
                    break
            
            # Cap at 1500 words - truncate at last sentence
            if word_count > 1500:
                words = summary.split()
                summary = ' '.join(words[:1500])
                # Try to end at a sentence boundary
                if not summary.endswith(('.', '!', '?')):
                    last_period = summary.rfind('.')
                    if last_period > 100:  # Ensure we don't cut too much
                        summary = summary[:last_period + 1]
        
        return summary
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
                    summary = summarize_text(raw_text, is_final=False)
                    word_count = count_words(summary)
                    st.success(f"Done ({word_count} words)")
                    all_summaries.append(summary)
                else:
                    st.warning(f"Could not extract text from: {url}")

            if all_summaries:
                st.divider()
                st.subheader("🧠 Combined Master Summary")
                
                combined = " ".join(all_summaries)
                final_summary = summarize_text(combined, is_final=True)
                final_word_count = count_words(final_summary)
                
                st.info(f"Word Count: {final_word_count} words")
                
                # Show warning if still outside range
                if final_word_count < 200:
                    st.warning(f"⚠️ Summary has only {final_word_count} words. Minimum is 200.")
                elif final_word_count > 1500:
                    st.warning(f"⚠️ Summary has {final_word_count} words. Maximum is 1500.")
                else:
                    st.success(f"✅ Summary meets requirements ({final_word_count} words)")
                
                st.write(final_summary)
                
                st.download_button("Download Text", final_summary, "summary.txt", "text/plain")
            else:
                st.error("No valid text could be extracted from the URLs.")
