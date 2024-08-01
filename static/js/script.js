const sampleQuestions = [
    "Tell me about yourself",
    "What is your name?",
    "What is your field of study?",
    "What motivated you to study computer science?",
    "Where have you worked previously?",
    "How do you approach teamwork and collaboration?",
    "What is the most exciting development in AI right now?",
    "What are your hobbies and interests?",
    "What is the best way to reach you?",
    "What programming languages are you proficient in?",
    "What are your primary responsibilities in your current job?",
    "How do you manage your time and prioritize tasks?",
    "How did you get started in the field of AI?",
    "Can you share any relevant projects or accomplishments you're particularly proud of?",
    "What motivates you to work in AI?",
    "What are your greatest strengths and weaknesses?",
    "How do you handle tight deadlines?"
];

function getRandomQuestions(n) {
    const shuffled = sampleQuestions.sort(() => 0.5 - Math.random());
    return shuffled.slice(0, n);
}

function updateSampleQuestions() {
    const container = document.getElementById('sampleQuestionContainer');
    const oldButtons = container.querySelectorAll('.sample-question');
    
    // Fade out old buttons
    oldButtons.forEach(button => {
        button.classList.add('fade-out');
    });

    // Wait for fade out, then update
    setTimeout(() => {
        container.innerHTML = '';
        const questions = getRandomQuestions(2);
        questions.forEach(question => {
            const button = document.createElement('button');
            button.className = 'sample-question fade-out';
            button.textContent = question;
            button.onclick = () => askQuestion(question);
            container.appendChild(button);
            
            // Trigger reflow
            button.offsetHeight;
            
            // Fade in new button
            setTimeout(() => {
                button.classList.remove('fade-out');
            }, 50); // Small delay to ensure the fade-in effect is visible
        });
    }, 300); // This should match the fade-out transition time in CSS
}

function askQuestion(question) {
    document.getElementById('userInput').value = question;
    sendMessage();
}

function sendMessage() {
    var userInput = document.getElementById('userInput');
    var message = userInput.value;
    if (message.trim() === '') return;
    
    fetch('/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({message: message}),
    })
    .then(response => response.json())
    .then(data => {
        addMessage('You', message, data.sentiment);
        addMessage('Simeon', data.response);
    });

    userInput.value = '';
}

function addMessage(sender, message, sentiment = null) {
    var chatMessages = document.getElementById('chatMessages');
    var messageElement = document.createElement('div');
    messageElement.className = 'message ' + (sender === 'You' ? 'user-message' : 'bot-message');
    
    if (sender === 'You' && sentiment) {
        var sentimentElement = document.createElement('span');
        sentimentElement.className = 'sentiment ' + sentiment;
        sentimentElement.textContent = sentiment;
        messageElement.appendChild(sentimentElement);
    }
    
    var textElement = document.createElement('span');
    textElement.className = 'message-text';
    messageElement.appendChild(textElement);
    
    chatMessages.appendChild(messageElement);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    if (sender === 'Simeon') {
        typeMessage(textElement, sender + ': ' + message);
    } else {
        textElement.textContent = sender + ': ' + message;
    }
}

function typeMessage(element, message) {
    var i = 0;
    var speed = 20; // milliseconds per character
    function type() {
        if (i < message.length) {
            element.textContent += message.charAt(i);
            i++;
            setTimeout(type, speed);
        }
    }
    type();
}

document.getElementById('userInput').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

// Update sample questions every 3 seconds
setInterval(updateSampleQuestions, 7000);

// Initial update of sample questions
updateSampleQuestions();