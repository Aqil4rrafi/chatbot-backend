import random
import json
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from model import ChatNet
from utils import bag_of_words, tokenize

app = Flask(__name__)
CORS(app)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('data/intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE, map_location=device)

model = ChatNet(data["input_size"], data["hidden_size"], data["output_size"]).to(device)
model.load_state_dict(data["model_state"])
model.eval()

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"answer": "Kirim pesan dong..."})
    
    sentence = tokenize(user_input)
    X = bag_of_words(sentence, data['all_words'])
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = data['tags'][predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return jsonify({"answer": random.choice(intent['responses'])})
    
    return jsonify({"answer": "Maaf, saya tidak mengerti pertanyaan itu."})

if __name__ == "__main__":
    app.run(port=5000, debug=True)