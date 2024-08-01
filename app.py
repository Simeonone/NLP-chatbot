from flask import Flask, render_template, request, jsonify
from chatbot import chatbot_response

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    result = chatbot_response(user_message)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)