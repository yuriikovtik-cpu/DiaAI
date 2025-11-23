from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
data = pd.read_csv("data/dataset.csv", sep=",", skiprows=1)
data = data.sample(frac=1, random_state=4).reset_index(drop=True)
train_data = data.iloc[:int(len(data))]

services = ['ЖЕК / КП з утримання житлового фонду', 'ОСББ (об’єднання співвласників)', 'Міськсвітло / Служба вуличного освітлення', 'Водоканал', 'Служба водовідведення / Каналізаційна служба', 'Теплокомуненерго / Міське тепло'
    , 'Газова служба / Аварійна газова служба', 'Електромережі / Енергетична компанія', 'Ліфтове господарство / Служба ліфтів', 'Департамент транспорту / Дорожній департамент', 'Управління благоустрою / КП Благоустрій'
    , 'Служба зеленого господарства / Департамент екології', 'Пожежна служба / ДСНС', 'Поліція / Муніципальна варта', 'Санепідемслужба / Департамент охорони здоров’я', 'Департамент освіти / охорони здоров’я (міський)'
    , 'Контакт-центр міської ради', 'Приватні підрядники (вивіз сміття, водії, технічне обслуговування)']
numbers = ["+380983302912", "+380503130117", "+380342789443", "+380800301586", "+380800301586", "+380342563511", "+380342591104", "+380800504020", "+380505853928", "+380342551913"
    , "+380342532212", "+380342759281", "+380342596507", "+380342792785", "+380342752878", "+380342535668", "+380800301586", "+380982745038"]
texts = train_data.iloc[:, 0].values
labels = train_data.iloc[:, 1].values

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts).toarray()
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.long)



class Classifier(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.fc = nn.Linear(inp, 64)
        self.fc2 = nn.Linear(64, out)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        return self.fc2(x)

model = Classifier(X.shape[1], 18)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


for epoch in range(300):
    preds = model(X)
    loss = loss_fn(preds, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def classify_query(query):
    vec = vectorizer.transform([query]).toarray()
    vec = torch.tensor(vec, dtype=torch.float32)
    probs = torch.softmax(model(vec),dim=1)
    pred = probs.argmax().item()
    return pred

