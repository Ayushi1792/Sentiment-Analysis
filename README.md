# WEB ARTICLE SENTIMENT ANALYZER

In today's world, the digital content is growing at an exponential rate making it difficult to interpret and process key information quickly. This project aims to solve this challenge by introducing intelligent machines that extract and analyzes useful content from user-provided web articles to deliver real time insights.

## BACKGROUND AND SCOPE

With the rise in AI-driven applications there's a need to automate the process of interpreting online content for public opinion, trends and context. This project leverages pre-trained Transformer models to analyze sentiment, extract summaries and draw key-insights.

Scope:
1. Supports any publicly accessible website URL
2. Performs Sentiment Analysis, Summarization and Named Entity Recognition(NER)
3. Displays the results via an Interactive Streamlit Dashboard deployed locally with an additional option to download data in JSON format.

This models helps students, researchers, and content analysts extract key information faster, saving time and resources. This also helps demonstrate the power of modern NLP for real-world text analytics.

## Getting Started

### Dependencies

* Python >= 3.9 recommended

* pip >= 22.0

* Operating System:

1. Windows 10 / Windows 11

2. macOS and Linux should also work (untested)

* Libraries:

1. streamlit

2. requests

3. beautifulsoup4

4. transformers

5. torch

### Installing

1. Clone the repository:
```
git clone https://github.com/Ayushi1792/Sentiment-Analysis.git
```
2. Navigate to the project folder: 
```
cd Sentiment-Analysis
```
3. Create a virtual environment(venv)[Recommended]: 
```
python -m venv .venv
```

4. Activate the virtual environment:

```
.venv\Scripts\activate (on Windows)

source .venv/bin/activate (on MacOs/ Linux)
```
5. Install required dependencies:

```
pip install -r requirements.txt
```
### Executing program

* Activate your virtual environment (if not already activated):

```
.venv\Scripts\activate
```
* Start Streamlit:

```
streamlit run app.py
```

* The app will open automatically in your browser.

* Paste any news article URL in the text box.

* Click Analyze.

* View sentiment, summary, and named entities.

## Help

* Ensure your transformers and torch packages are up-to-date.
```
pip install --upgrade transformers torch
```
* If the models fail to load:
```
pip install --upgrade pip
```
* Restart virtual environment if necessary.
## Authors

Ayushi Ojha
GitHub: @Ayushi1792

## Version History

* 0.2
    * Added entity merging to clean NER results
    * Improved error handling and UI

* 0.1
    * Initial release with sentiment, summarization, and NER

## License

This project is licensed under the MIT License - see the LICENSE.md file for details

## Acknowledgments

* Hugging Face Transformers – for the NLP models

* Streamlit – for the UI framework

* BeautifulSoup – for HTML parsing

* awesome-readme

* PurpleBooth

* dbader

* zenorocha

