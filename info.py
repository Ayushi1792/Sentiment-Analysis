import json
import requests
import streamlit as st
from bs4 import BeautifulSoup
from collections import Counter
from transformers import pipeline

# 1 - Initialize NLP pipelines
@st.cache_resource
def load_pipelines():
    sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=-1)
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1, model_kwargs={"torch_dtype": "float32"})
    ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple", model_kwargs={"torch_dtype": "float32"})
    return sentiment, summarizer, ner

sentiment_pipeline, summarizer_pipeline, ner_pipeline = load_pipelines()

# 2 - Helper function for text chunking
def chunk_text(text, chunk_size=512):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# 3 - Clean unwanted lines
def clean_text(raw_text):
    lines = raw_text.split('\n')
    filtered = [line.strip() for line in lines if len(line.strip()) > 30 and "reddit" not in line.lower()]
    return "\n".join(filtered)

# 4 - Extract article content using requests and BeautifulSoup
def extract_text_from_url(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, timeout=10, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")

        article = soup.find("article")
        paragraphs = article.find_all("p") if article else soup.find_all("p")

        filtered_paragraphs = [
            p.get_text(strip=True) for p in paragraphs
            if len(p.get_text(strip=True)) > 50 and "samaritans" not in p.get_text(strip=True).lower()
        ]

        return clean_text(" ".join(filtered_paragraphs))
    except Exception as e:
        st.error(f"Failed to fetch content: {e}")
        return ""

# 5 - Analyze a URL
def analyze_url(url):
    text = extract_text_from_url(url)
    if not text:
        return None

    chunks = chunk_text(text)
    sentiments = [sentiment_pipeline(chunk)[0] for chunk in chunks]
    sentiment_counts = Counter([s["label"] for s in sentiments])
    overall_sentiment = max(sentiment_counts, key=sentiment_counts.get)

    summary = summarizer_pipeline(text[:1024])[0]['summary_text']
    entities = ner_pipeline(text[:512])
    top_entities = Counter([e['word'] for e in entities]).most_common(10)

    return {
        "url": url,
        "summary": summary,
        "overall_sentiment": overall_sentiment,
        "sentiment_counts": dict(sentiment_counts),
        "sentiment_chunks": sentiments,
        "entities": entities,
        "top_entities": top_entities
    }

# 6 - Streamlit App
st.set_page_config(page_title="Sentiment Analyzer", layout="wide")
st.title("🌐 Web Article Sentiment Analyzer")
st.markdown("Analyze sentiment, summary, and named entities from any news article.")

url_input = st.text_input("Enter Article URL", placeholder="https://www.bbc.com/news/...")
if st.button("Analyze") and url_input:
    with st.spinner("Analyzing... please wait ⏳"):
        try:
            result = analyze_url(url_input)
            if result is None:
                st.error("Could not extract or analyze the article.")
            else:
                st.success("✅ Analysis Complete!")

                # Display Summary
                st.subheader("📄 Summary")
                st.write(result["summary"])

                # Display Overall Sentiment
                st.subheader("🧠 Overall Sentiment")
                st.metric(label="Sentiment", value=result["overall_sentiment"])
                st.write("**Sentiment Breakdown:**", result["sentiment_counts"])

                # Named Entities
                st.subheader("🔍 Top Named Entities")
                for word, count in result["top_entities"]:
                    st.write(f"- **{word}**: {count} time(s)")

                # Download JSON
                st.subheader("📥 Download Results as JSON")
                st.download_button(
                    label="Download Result",
                    data=json.dumps(result, indent=2, ensure_ascii=False, default=str),
                    file_name="analysis_result.json",
                    mime="application/json"
                )

        except Exception as e:
            st.error(f"❌ Error: {e}")
