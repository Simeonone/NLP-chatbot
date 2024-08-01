import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from textblob import TextBlob
from transformers import AutoTokenizer, AutoModel
import torch
from typing import Optional
import random

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load Hugging Face model
print("Loading Hugging Face model...")
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
print("Hugging Face model loaded.")

# Load spaCy model for NER
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")
print("spaCy model loaded.")

# Dictionary to store question-answer pairs
qa_pairs = {}

# Function to extract questions and answers
def extract_qa(text):
    lines = text.split('\n')
    current_question = ""
    current_answer = ""
    for line in lines:
        if re.match(r'^\d+\.', line):
            if current_question and current_answer:
                qa_pairs[current_question.strip()] = current_answer.strip()
            current_question = line
            current_answer = ""
        else:
            current_answer += line + " "
    if current_question and current_answer:
        qa_pairs[current_question.strip()] = current_answer.strip()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    return ' '.join(tokens)

def get_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return "positive"
    elif analysis.sentiment.polarity < 0:
        return "negative"
    else:
        return "neutral"

def text_to_vector(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def semantic_similarity(query, questions):
    query_vector = text_to_vector(query)
    question_vectors = [text_to_vector(q) for q in questions]
    similarities = cosine_similarity([query_vector], question_vectors)[0]
    best_match_index = np.argmax(similarities)
    return questions[best_match_index], similarities[best_match_index]

GREETING_RESPONSES = {
    r'\b(hi|hey|hello)\b': "Hello! I'm Simeon's AI assistant. How can I help you with information about my background or skills? You can choose from any of the sample questions below",
    r'\bgreetings\b': "Greetings! I'm here to provide information about Simeon. What would you like to know about me? Feel free to select one of the sample questions below",
    r'\bgood (morning|afternoon|evening)\b': "Good {time_of_day}! I am Simeon's chatbot. Feel free to ask me any question regarding my professional or education background. Or select one of the sample questions below"
}

def get_greeting_response(user_input: str) -> Optional[str]:
    user_input_lower = user_input.lower()
    for pattern, response in GREETING_RESPONSES.items():
        match = re.search(pattern, user_input_lower)
        if match:
            if 'time_of_day' in response:
                return response.format(time_of_day=match.group(1))
            return response
    return None

fallback_responses = [
    "I'm not sure I understand. Could you please rephrase your question?",
    "I don't have specific information about that. Could you please rephrase your question or ask something else about my background or skills?",
    "I'm afraid I don't have an answer for that. Is there anything else you'd like to know about my education or work experience?",
]

def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def match_question(user_input):
    preprocessed_input = preprocess_text(user_input)
    questions = list(qa_pairs.keys())
    best_match, similarity = semantic_similarity(preprocessed_input, questions)
    
    # Print top 3 matches for debugging
    similarities = [semantic_similarity(preprocessed_input, [q])[1] for q in questions]
    top_3 = sorted(zip(questions, similarities), key=lambda x: x[1], reverse=True)[:3]
    for q, s in top_3:
        print(f"  {s:.4f}: {q}")
    
    if similarity > 0.3:  # Adjust threshold as needed
        return best_match
    else:
        return None

def chatbot_response(user_input):
    greeting_response = get_greeting_response(user_input)
    if greeting_response:
        return {"response": greeting_response, "sentiment": "positive"}
    sentiment = get_sentiment(user_input)
    entities = extract_entities(user_input)
    matched_question = match_question(user_input)
    
    if matched_question:
        response = qa_pairs[matched_question]
    else:
        response = random.choice(fallback_responses)
    return {"response": response, "sentiment": sentiment}
document_content = """
1. What is your full name?
Simeon Kengere Osiemo

2. Can you tell me about yourself?
Well, I'm Simeon Kengere Osiemo, a computer scientist with a passion for AI that borders on obsession. I'm the kind of person who dreams in Python and wakes up thinking about machine learning algorithms. By day, I'm a software engineer at the Ministry of Interior and National Administration, where I spend my time making digital systems more secure and efficient. I've worn many hats in my career - from data scientist to systems integration engineer - but my favorite is my invisible 'AI wizard' hat. I've developed everything from handwritten digit recognition systems to fall detection devices for the elderly. You could say I'm on a mission to make machines as smart as possible, while keeping my own wits sharp enough to stay ahead of them.
When I'm not coding or studying, you might find me writing technical blog posts with titles like 'How math makes machines intelligent: The magic behind AI.' It's my way of spreading the AI gospel to the masses. And yes, I'm proud to say I'm a card-carrying member of the IEEE. In short, I'm a techie with a sense of humor, always ready to tackle the next big challenge in the world of AI and software engineering. Just don't ask me to fix your printer - that's one problem even AI can't solve!

3. Where did you complete your undergraduate degree?
The University of Nairobi

4. Where did you complete your master’s degree?
The University of Nairobi

5. What is your field of study?
Computer Science, specialized in Computational Intelligence

6. What motivated you to study computer science?
My fascination with computer science began when I realized I could create entire worlds with just logic and code. I love the problem-solving aspect, where each project is a new puzzle to crack. Plus, in a rapidly developing country like Kenya, I saw how technology could leapfrog traditional challenges. The constant evolution in this field keeps me excited - there's always something new to learn. Ultimately, computer science offers me the perfect blend of creativity, problem-solving, and the chance to make a real impact. And who knows? Maybe one day I'll create that benevolent AI overlord we've all been waiting for!

7. What specific courses have you taken related to AI and machine learning?
Master of Science in Computer Science, Bachelor of Science in Computer Science, Diploma in Computer Science, how to build a generative A.I project from AWS, and Python for Data Science from IBM

8. Where have you worked previously?
I have worked at Ongata Rongai sub County hospital, AfyaPro 2.0 Connected care, Fujita Corporation, The University of Nairobi, and The Ministry of Interior and National Administration

9. What roles have you held in the past?
Software engineer, Data Scientist, Systems Integration Engineer and Research Assistant

10. How do you stay current with new technologies and trends in AI?
Staying current with new technologies and trends in AI is essential, and I do so through continuous learning and community engagement. I regularly take online courses on platforms like Udemy and attend webinars and workshops hosted by industry leaders like AWS. I also read academic papers from conferences like NeurIPS and follow AI research on arXiv. Engaging with the community through conferences, meetups, and online forums like Reddit, Medium and LinkedIn keeps me connected with peers and industry experts. Additionally, I work on personal projects to apply new techniques and contribute to open-source projects on GitHub. Lastly, I stay informed by following tech news websites such as TechCrunch and VentureBeat. This multifaceted approach ensures I remain at the forefront of AI advancements.

11. What programming languages are you proficient in?
Python, SQL and JavaScript

12. Can you describe your current job?
As a Software Engineer at the Ministry of Interior and National Administration, my role involves developing and deploying secure, scalable digital record management systems. By adhering to clean code practices, I’ve been able to enhance system efficiency by 45% and significantly reduce potential security vulnerabilities by 30%. I collaborate closely with cross-functional teams, leveraging Agile methodologies to address and resolve critical system vulnerabilities. This collaborative approach ensures system stability and maintains the integrity of our data. Additionally, I focus on refactoring and optimizing internal web applications using both Python and JavaScript. This optimization has led to a 25% improvement in response times and has greatly enhanced the user experience for over 500 ministry staff members. Overall, my work is centered on maintaining high standards of security, efficiency, and user satisfaction.

13. What are your primary responsibilities in your current job?
As a Software Engineer at the Ministry of Interior and National Administration, my primary responsibilities include developing and deploying secure, scalable digital record management systems, identifying and mitigating security vulnerabilities, and collaborating with cross-functional teams using Agile methodologies to ensure system stability. I also refactor and optimize internal web applications with Python and JavaScript, enhancing response times and user experience for ministry staff, while maintaining the integrity and accuracy of our data.

14. What's your background and experience in AI? I have a strong background in AI, supported by a Master's degree in Computer Science and extensive hands-on experience. I've developed machine learning models, data-driven applications, and AI-based solutions for various projects, including handwritten digit recognition, real-time fall detection for elderly care, and precision farming. Proficient in Python and TensorFlow, I've implemented scalable AI solutions and collaborated on research projects, while using my expertise in AI, machine learning, and natural language processing.

15. How did you get started in the field of AI?
I got started in AI during my undergraduate studies in Computer Science at the University of Nairobi, where I was introduced to machine learning and data science concepts. My interest deepened as I engaged in various projects and research that applied AI techniques to solve real-world problems, such as developing a handwritten digit recognition system and a precision farming support system. This foundation led me to pursue a Master's degree in Computer Science, focusing on AI, and gaining practical experience through roles that involved AI-driven projects.

16. What motivates you to work in AI?
I am motivated to work in AI because of its transformative potential to solve complex problems and improve lives. The ability to create intelligent systems that can learn, adapt, and provide insights inspires me to innovate and push the boundaries of technology. I am particularly driven by the impact AI can have in various fields, from healthcare to agriculture, and the opportunity to contribute to advancements that can benefit society as a whole.

17. Are you willing to relocate?
Yes

18. What are your greatest strengths and weaknesses?
My greatest strengths include my strong problem-solving skills and my ability to quickly learn and adapt to new technologies. I am proficient in various programming languages and AI tools, which allows me to develop effective solutions and optimize performance. I also have excellent teamwork and communication skills, enabling me to collaborate effectively with cross-functional teams.
As for weaknesses, I sometimes tend to be overly detail-oriented, which can slow down progress. However, I am actively working on finding a balance between attention to detail and maintaining efficiency to ensure timely project completion.

19. Can you share any relevant projects or accomplishments you're particularly proud of?
I am currently developing an AI based chatbot to be integrated into the the E-Citizen facebook messenger to handle any queries that Kenyan citizens might have regarding e-citizen. In addition, I am particularly proud of developing a precision farming support system using AI and machine vision to identify defective areas in farmland, which can significantly improved agricultural efficiency. Another notable project is a handwritten digit recognition system utilizing convolutional neural networks, which demonstrated high accuracy and showcased my skills in AI and machine learning. Additionally, my work on a real-time fall detection system for elderly care, combining AI with embedded devices, stands out as it has the potential to enhance safety and response times for vulnerable populations.


20. Can you describe a situation where you had to communicate technical concepts to a non-technical audience?
In my role at the Ministry of Interior and National Administration, I had to present the benefits and functionality of our new digital record management system to senior officials who were not familiar with technical jargon. I used simple analogies and visual aids to explain how the system improved efficiency and security. By focusing on the practical benefits and addressing their concerns in layman's terms, I successfully conveyed the technical concepts and gained their support for the project.

21. Can you show me your GitHub profile?
It is https://github.com/Simeonone

22. Can you list some of the open-source AI tools you have used?
I have used several open-source AI tools, including TensorFlow, Keras, and PyTorch for building and training machine learning models, Scikit-learn for data analysis and machine learning tasks, and OpenCV for computer vision projects. Additionally, I have worked with NLTK for natural language processing and NumPy and Pandas for data manipulation and analysis.

23. What machine learning frameworks are you familiar with?
I am familiar with a variety of machine learning frameworks, including TensorFlow, Keras, PyTorch, Scikit-learn, Edge Impulse and XGBoost.

24. How do you approach problem-solving in AI projects?
When approaching problem-solving in AI projects, I start by thoroughly understanding the problem and its context. I gather and preprocess relevant data, ensuring its quality and suitability for the task. I then select appropriate models and algorithms, often experimenting with several to find the best fit. I iteratively train and validate these models, tuning hyperparameters to optimize performance. Throughout the process, I use clear metrics to evaluate success and make data-driven decisions. Finally, I ensure that the solution is scalable, interpretable, and meets the stakeholders' requirements, continually monitoring and refining it as needed.

25. How can I contact you?
You can reach me by mobile on +254713336627 or by email on simeon.kengere@gmail.com

26. What techniques do you use for feature engineering?
I employ various techniques for feature engineering, including feature selection, extraction, and transformation. For feature selection, I use methods such as recursive feature elimination and mutual information. For feature extraction, I leverage tools like Principal Component Analysis (PCA) and t-Distributed Stochastic Neighbor Embedding (t-SNE) to reduce dimensionality while retaining essential information. Additionally, I perform feature transformation techniques like normalization and standardization to prepare the data for model training effectively

27. How do you manage your time and prioritize tasks?
I manage my time and prioritize tasks using a combination of tools and strategies. I rely on project management tools like Jira and Trello to organize tasks and track progress. I also use the Eisenhower Matrix to categorize tasks based on urgency and importance, ensuring that high-priority tasks are addressed first. Additionally, I break down larger projects into smaller, manageable tasks and set specific deadlines to stay on track. Regularly reviewing my task list and adjusting priorities as needed helps me maintain productivity and meet deadlines efficiently.

28. What is the best way to reach you?
You can reach me by mobile on +254713336627 or by email on simeon.kengere@gmail.com

29. How do you handle tight deadlines?
I handle tight deadlines by prioritizing tasks, breaking them down into manageable steps, and maintaining clear communication. I use project management tools to track progress and ensure that I stay on schedule. Effective time management techniques, such as the Pomodoro Technique, help me stay focused and productive. Additionally, I remain flexible and adaptable, ready to adjust plans as needed to meet the deadline. Clear communication with team members and stakeholders ensures that everyone is aligned and any potential issues are addressed promptly.

30. What are your hobbies and interests?
I enjoy programming and developing side projects to explore new technologies. Outside of the tech world, I like hiking, playing chess, and experimenting with new cooking recipes.

31. Can you give an example of how you resolved a conflict in a professional setting?
In a previous role, there was a conflict between team members regarding the approach to a project. One group preferred a traditional method while the other wanted to experiment with a new technique. I facilitated a meeting where each side could present their arguments and concerns. We then discussed the pros and cons of each approach and ultimately decided on a hybrid solution that incorporated elements of both methods. This not only resolved the conflict but also resulted in a more innovative and effective solution.

32. How do you approach teamwork and collaboration?
I approach teamwork and collaboration with a focus on clear communication, mutual respect, and leveraging the strengths of each team member. I believe in setting clear goals and expectations from the outset and maintaining open lines of communication throughout the project. I also value diversity of thought and encourage team members to share their ideas and perspectives. Regular check-ins and feedback sessions help ensure that everyone is aligned and working towards the common goal.

33. What is the most exciting development in AI right now?
One of the most exciting developments in AI right now is the advancement of large language models, such as GPT-4. These models are pushing the boundaries of natural language understanding and generation, enabling more sophisticated and human-like interactions.

34. Where did you study?
The University of Nairobi
"""


# Extract question-answer pairs
extract_qa(document_content)

# Start the chatbot
if __name__ == "__main__":
    # chatbot()
    pass