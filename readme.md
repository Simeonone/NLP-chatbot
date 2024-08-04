
# Simeon's NLP Chatbot

An interactive NLP-powered chatbot that simulates a conversation with Simeon, showcasing his AI expertise through dynamic responses, sentiment analysis, and a user-friendly interface.

## Project Description
This project is an interactive Natural Language Processing (NLP) chatbot that simulates a conversation with Simeon, showcasing his background, skills, and experiences in the field of AI and computer science. The chatbot uses advanced NLP techniques to understand user queries and provide relevant responses.

## Features
- Natural language understanding and processing
- Sentiment analysis of user input
- Dynamic response generation
- Efficient semantic similarity matching
- Named Entity Recognition (NER)
- Greeting handling
- Fallback responses for unmatched queries

## Technologies Used
- Backend:
  - Python
  - Flask (Web framework)
  - NLTK (Natural Language Toolkit)
  - TextBlob (for sentiment analysis)
  - spaCy (for named entity recognition)
  - Hugging Face Transformers (for semantic similarity)
  - PyTorch
- Frontend:
  - HTML5
  - CSS3
  - JavaScript (ES6+)
- NLP Model:
  - Sentence Transformer (all-MiniLM-L6-v2)

## Setup and Installation
1. Clone the repository:

`git clone [https://github.com/Simeonone/NLP-chatbot.git](https://github.com/Simeonone/NLP-chatbot.git)` 

`cd NLP-chatbot`

2. Set up a virtual environment (optional but recommended):

`python -m venv venv source venv/bin/activate` # On Windows use `venv\Scripts\activate`

3. Install the required packages:

`pip install -r requirements.txt`

4. Download necessary NLTK data: `python`  `import nltk`  `nltk.download('punkt')`  `nltk.download('stopwords')`

5.  Download the spaCy English model:
    `python -m spacy download en_core_web_sm`
    
6.  Run the Flask application:
    `python app.py`
    
7.  Open a web browser and navigate to `http://localhost:5000`

## Usage

-   Type your questions in the input field.
-   The chatbot will analyze the sentiment of your input and provide a relevant response.
-   Explore various aspects of Simeon's background, skills, and experiences in AI and computer science.

## Project Structure

![image](https://github.com/user-attachments/assets/b3524867-1b99-4647-8fb0-5adaf7e7f8d5)


## Customization

-   To modify the chatbot's responses, edit the `document_content` in the `chatbot.py` file.
-   Adjust the greeting responses by modifying the `GREETING_RESPONSES` dictionary in `chatbot.py`.
-   Fine-tune the semantic similarity threshold in the `match_question` function in `chatbot.py`.
-   Modify the fallback responses by updating the `fallback_responses` list in `chatbot.py`.

## Contributing

Contributions to improve the chatbot are welcome. Please follow these steps:

1.  Fork the repository
2.  Create a new branch (`git checkout -b feature-branch`)
3.  Make your changes and commit them (`git commit -am 'Add some feature'`)
4.  Push to the branch (`git push origin feature-branch`)
5.  Create a new Pull Request
