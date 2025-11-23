from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("data/dataset.csv", sep=",", skiprows=1)
data = data.sample(frac=1, random_state=4).reset_index(drop=True)
services = ['ЖЕК / КП з утримання житлового фонду', 'ОСББ (об’єднання співвласників)', 'Міськсвітло / Служба вуличного освітлення', 'Водоканал', 'Служба водовідведення / Каналізаційна служба', 'Теплокомуненерго / Міське тепло'
    , 'Газова служба / Аварійна газова служба', 'Електромережі / Енергетична компанія', 'Ліфтове господарство / Служба ліфтів', 'Департамент транспорту / Дорожній департамент', 'Управління благоустрою / КП Благоустрій'
    , 'Служба зеленого господарства / Департамент екології', 'Пожежна служба / ДСНС', 'Поліція / Муніципальна варта', 'Санепідемслужба / Департамент охорони здоров’я', 'Департамент освіти / охорони здоров’я (міський)'
    , 'Контакт-центр міської ради', 'Приватні підрядники (вивіз сміття, водії, технічне обслуговування)']
numbers = ["+380983302912", "+380503130117", "+380342789443", "+380800301586", "+380800301586", "+380342563511", "+380342591104", "+380800504020", "+380505853928", "+380342551913"
    , "+380342532212", "+380342759281", "+380342596507", "+380342792785", "+380342752878", "+380342535668", "+380800301586", "+380982745038"]

train_data = data.iloc[:int(len(data)*0.9)]
test_data = data.iloc[int(len(data)*0.1):]

texts = train_data.iloc[:,0].values
labels_raw = train_data.iloc[:,1].values

le = LabelEncoder()
y = torch.tensor(le.fit_transform(labels_raw), dtype=torch.long)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts).toarray()
X = torch.tensor(X, dtype=torch.float32)

class Classifier(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.fc = nn.Linear(inp, 64)
        self.fc2 = nn.Linear(64, out)
    def forward(self, x):
        x = torch.relu(self.fc(x))
        return self.fc2(x)

model = Classifier(X.shape[1], len(le.classes_))
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(300):
    preds = model(X)
    loss = loss_fn(preds, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# --- Класифікація ---
def classify_query(query: str) -> str:
    vec = vectorizer.transform([query]).toarray()
    vec = torch.tensor(vec, dtype=torch.float32)
    probs = torch.softmax(model(vec), dim=1)
    pred = probs.argmax().item()
    return pred

# --- Flask ---
app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"result": "Помилка: порожній текст"})
    predicter = classify_query(text)
    result = f"Ми отримали ваше звернення: '{text} '\nЗверніться до:  {services[predicter]}\n{numbers[predicter]}"
    return jsonify({"result": str(result)})

if __name__ == "__main__":
    app.run(debug=True)
