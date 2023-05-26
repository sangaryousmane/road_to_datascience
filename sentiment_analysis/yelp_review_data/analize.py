import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter

class SentimentDataset(Dataset):
    def __init__(self, data, labels, word2idx):
        self.data = data
        self.labels = labels
        self.word2idx = word2idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        data_indices = [self.word2idx[word] for word in data.split() if word in self.word2idx]
        data_tensor = torch.tensor(data_indices)
        label_tensor = torch.tensor(label)
        return data_tensor, label_tensor

class CNNBiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=hidden_dim, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=embedding_dim, out_channels=hidden_dim, kernel_size=8)
        self.lstm = nn.LSTM(input_size=hidden_dim*2, hidden_size=hidden_dim, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.permute(0, 2, 1)
        conv1_output = F.relu(self.conv1(embedded))
        conv2_output = F.relu(self.conv2(embedded))
        conv1_output = F.max_pool1d(conv1_output, conv1_output.shape[1]).squeeze()
        conv2_output = F.max_pool1d(conv2_output, conv2_output.shape[1]).squeeze()
        concatenated = torch.cat((conv1_output, conv2_output), dim=1)
        concatenated = concatenated.unsqueeze(0)
        lstm_output, _ = self.lstm(concatenated)
        lstm_output = lstm_output.squeeze()
        lstm_output = self.dropout(lstm_output)
        output = self.fc(lstm_output)
        return output

# Define hyperparameters
embedding_dim = 100
hidden_dim = 256
output_dim = 2
dropout = 0.5
lr = 0.001
batch_size = 32
num_epochs = 10


# Train and test
yelp_reviews = pd.read_csv('data/yelp.csv')
yelp_reviews.head(10)
train_data, test_data = train_test_split(yelp_reviews,
                                               test_size=0.2, random_state=42)
# Load data
train_data = train_data['text'].tolist();
train_labels = pd.read_csv('data/yelp.csv')['stars'].tolist()
word_counter = Counter()
for sentence in train_data:
    words = sentence.split()
    word_counter.update(words)
word2idx = {word: i for i, (word, count) in enumerate(word_counter.items()) if count >= 5}
train_dataset = SentimentDataset(train_data, train_labels, word2idx)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize model and optimizer
model = CNNBiLSTM(len(word2idx), embedding_dim, hidden_dim, output_dim, dropout)
optimizer = optim.Adam(model.parameters(), lr=lr)

# Train model
# for epoch in range(num_epochs):
#     for data, labels in train_loader:
#         optimizer.zero_grad()
#         output = model(data)
#         loss = F.cross_entropy(output, labels)
#         loss.backward()
#         optimizer.step()

# Evaluate model
test_data = train_data['text'].tolist();
test_labels = pd.read_csv('data/yelp.csv')['stars'].tolist()
test_dataset = SentimentDataset(test_data, test_labels, word2idx)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for data, labels in test_loader:
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print('Test accuracy: {:.2f}%'.format(accuracy))