import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

yelp_reviews = pd.read_csv('data/yelp.csv')
yelp_reviews.head(10)

# train_reviews, test_reviews = train_test_split(yelp_reviews, test_size=0.2, random_state=42)

# Splitting the data into training and test data
train_reviews, test_reviews = train_test_split(yelp_reviews, test_size=0.2, random_state=42)


# Define a class for the dataset
class YelpDataset(Dataset):

    def __init__(self, reviews, tokenizer):
        self.reviews = reviews
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = self.reviews.iloc[idx]['text']
        sentiment = self.reviews.iloc[idx]['sentiment']
        tokens = self.tokenizer.encode(review, add_special_tokens=True)
        return torch.tensor(tokens), torch.tensor(sentiment)


# Tokenize the reviews
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_dataset = YelpDataset(train_reviews, tokenizer)
test_dataset = YelpDataset(test_reviews, tokenizer)


# Define the CNN-BiLSTM model
class YelpModel(nn.Module):
    def __init__(self, num_classes):
        super(YelpModel, self).__init__()
        self.embedding = nn.Embedding(tokenizer.vocab_size, 128)
        self.conv = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.bilstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=2, bidirectional=True)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.pool(x)
        x = x.permute(2, 0, 1)
        x, _ = self.bilstm(x)
        x = x[-1, :, :]
        x = self.fc(x)
        return x


# Instantiate the model and define the loss function and optimizer
model = YelpModel(num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

for epoch in range(10):
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print('Epoch %d loss: %.3f' % (epoch + 1, running_loss / len(train_loader)))

# Evaluate the model on the test set
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Test accuracy: %.2f%%' % (100 * correct / total))
