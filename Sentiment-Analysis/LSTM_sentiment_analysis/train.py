import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torch.optim as optim
import pickle


# 모델 정의
class Model(nn.Module):
    def __init__(self, vocabs, embedding_size, tag_size, hidden_size, rnn_layers, dropout_rate):
        super(Model, self).__init__()

        word2id, tag2id = vocabs
        # word 임베딩
        self.word_embeddings = nn.Embedding(len(word2id), embedding_size)
        # TAG 임베딩
        self.tag_embeddings = nn.Embedding(len(tag2id), tag_size)

        self.lstm = nn.LSTM(embedding_size + tag_size, hidden_size, rnn_layers, batch_first=True, bidirectional=True)

        self.linear = nn.Linear(hidden_size * 2, 1)

        self.dropout_rate = dropout_rate

    def forward(self, x_word, x_tag):
        x_word_emb = self.word_embeddings(x_word)
        x_tag_emb = self.tag_embeddings(x_tag)

        rnn_input = torch.cat([x_word_emb, x_tag_emb], dim=-1)

        rnn_input = F.dropout(rnn_input, self.dropout_rate, self.training)

        outputs, (hn, cn) = self.lstm(rnn_input)

        output = torch.cat([hn[0], hn[1]], dim=-1)

        output = torch.sigmoid(self.linear(output)).squeeze(1)

        return output


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("data.pkl", "rb") as f:
        data = pickle.load(f)
    trainset, testset, word2id, tag2id = data['train'], data['test'], data['w2id'], data['t2id']

    model = Model((word2id, tag2id), embedding_size=300, tag_size=50, hidden_size=300, rnn_layers=1, dropout_rate=0.3)
    model.to(device)

    parameters = [p for p in model.parameters() if p.requires_grad]

    # Binary CrossEntropy Loss
    criterion = nn.BCELoss()

    optimizer = optim.Adam(parameters)

    batch_size = 128
    epochs = 20

    losses = []
    for epoch in range(1, epochs + 1):
        num_data = len(trainset)
        num_batch = (num_data + batch_size - 1) // batch_size

        # Train
        model.train()
        for ii in range(num_batch):
            start = ii * batch_size
            end = num_data if (ii + 1) * batch_size > num_data else (ii + 1) * batch_size

            batch_data = trainset[start:end]

            batch_word_ids = [torch.tensor(data[0], dtype=torch.long) for data in batch_data]
            batch_tag_ids = [torch.tensor(data[1], dtype=torch.long) for data in batch_data]
            batch_labels_ids = [data[2] for data in batch_data]

            batch_word_ids = pad_sequence(batch_word_ids, batch_first=True)
            batch_tag_ids = pad_sequence(batch_tag_ids, batch_first=True)

            batch_labels_ids = torch.tensor(batch_labels_ids, dtype=torch.float)

            batch_word_ids = batch_word_ids.to(device)
            batch_tag_ids = batch_tag_ids.to(device)
            batch_labels_ids = batch_labels_ids.to(device)

            batch_outputs = model(batch_word_ids, batch_tag_ids)

            loss = criterion(batch_outputs, batch_labels_ids)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            losses.append(loss.data)

            if (ii + 1) % 200 == 0:
                print("%6d/%6d: loss %.6f" % (ii + 1, num_batch, sum(losses) / len(losses)))
                losses = []

        # Eval
        num_data = len(testset)
        num_batch = (num_data + batch_size - 1) // batch_size

        model.eval()

        total = len(testset)
        match = 0
        for ii in range(num_batch):
            start = ii * batch_size
            end = num_data if (ii + 1) * batch_size > num_data else (ii + 1) * batch_size

            batch_data = testset[start:end]

            batch_word_ids = [torch.tensor(data[0], dtype=torch.long) for data in batch_data]
            batch_tag_ids = [torch.tensor(data[1], dtype=torch.long) for data in batch_data]
            batch_labels_ids = [data[2] for data in batch_data]

            batch_word_ids = pad_sequence(batch_word_ids, batch_first=True)
            batch_tag_ids = pad_sequence(batch_tag_ids, batch_first=True)

            batch_word_ids = batch_word_ids.to(device)
            batch_tag_ids = batch_tag_ids.to(device)

            batch_outputs = model(batch_word_ids, batch_tag_ids)

            batch_outputs = batch_outputs.data.cpu().numpy().tolist()

            batch_pred_ids = [1 if output >= 0.5 else 0 for output in batch_outputs]

            for a, o in zip(batch_labels_ids, batch_pred_ids):
                if a == o:
                    match += 1

        print("Epoch %d, match : %6d, total : %6d, ACC : %.2f" % (epoch, match, total, 100 * match / total))


if __name__ == "__main__":
    train()
