import torch
from model import ChatNet
from utils import bag_of_words, tokenize

# Load hasil training
data = torch.load("data.pth")

# Siapkan model dengan ukuran yang sesuai hasil training
model = ChatNet(data["input_size"], data["hidden_size"], data["output_size"])
model.load_state_dict(data["model_state"])
model.eval() # Set ke mode pakai (bukan belajar)

print("Bot siap! (ketik 'quit' untuk berhenti)")
while True:
    msg = input("Kamu: ")
    if msg == "quit": break
    
    # Ubah input kamu jadi angka seperti saat training
    tokens = tokenize(msg)
    X = bag_of_words(tokens, data['all_words'])
    X = torch.from_numpy(X.reshape(1, X.shape[0]))

    # Minta model menebak
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = data['tags'][predicted.item()]
    
    print(f"Bot: [Mendeteksi Tag: {tag}]")