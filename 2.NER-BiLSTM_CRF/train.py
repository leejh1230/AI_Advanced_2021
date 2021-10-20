import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
from torchcrf import CRF
from utils import batchify, evaluate_ner_F1, evaluate_ner_F1_and_write_result
import os


class Model(nn.Module):
    def __init__(self, vocabs, word_dim, pos_dim, hidden_size, rnn_layers, dropout_rate,
                 bidirectional=True, use_crf=False, embedding=None):
        super(Model, self).__init__()

        word2id, tag2id, label2id = vocabs
        self.word_embeddings = nn.Embedding(len(word2id), word_dim)
        if embedding is not None:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(embedding))

        self.tag_embeddings = nn.Embedding(len(tag2id), pos_dim)

        self.lstm = nn.LSTM(word_dim + pos_dim, hidden_size, rnn_layers,
                            batch_first=True, bidirectional=bidirectional, dropout=dropout_rate)

        output_size = hidden_size * 2 if bidirectional else hidden_size

        self.linear = nn.Linear(output_size, len(label2id))

        self.dropout_rate = dropout_rate

        self.use_crf = use_crf
        if use_crf:
            self.crf = CRF(len(label2id), batch_first=True)

        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

    def forward(self, word_ids, tag_ids, label_ids):
        word_emb = self.word_embeddings(word_ids)
        tag_emb = self.tag_embeddings(tag_ids)

        rnn_input = torch.cat([word_emb, tag_emb], dim=-1)

        rnn_input = F.dropout(rnn_input, self.dropout_rate, self.training)

        rnn_outputs, (hn, cn) = self.lstm(rnn_input)

        logits = self.linear(rnn_outputs)

        # [1, 1, 1, 0, 0]
        # [1, 1, 1, 1, 1]
        mask = word_ids.ne(0)
        if self.training:  # training
            if self.use_crf:
                loss = -self.crf(logits, label_ids, mask=mask.byte())
                return loss

            else:
                batch, seq_len, num_label = logits.size()

                logits = logits.view(-1, logits.data.shape[-1])
                label_ids = label_ids.view(-1)

                loss = F.cross_entropy(logits, label_ids, reduction='none')
                loss = loss.view(batch, seq_len)

                loss = loss * mask.float()

                num_tokens = mask.sum(1).sum(0)

                loss = loss.sum(1).sum(0) / num_tokens
                return loss

        label_ids = label_ids.data.cpu().numpy().tolist()
        lengths = mask.sum(1).long().tolist()

        answers = []
        for answer, length in zip(label_ids, lengths):
            answers.append(answer[:length])

        if self.use_crf:
            predictions = self.crf.decode(logits, mask)

            return answers, predictions

        batch_preds = torch.argmax(logits, dim=-1)
        batch_preds = batch_preds.data.cpu().numpy().tolist()

        predictions = []
        for pred, length in zip(batch_preds, lengths):
            predictions.append(pred[:length])

        return answers, predictions


def train(epochs=30, batch_size=32,
          word_dim=100, pos_dim=50, hidden_size=300, rnn_layers=1, bidirectional=False,
          use_pretrained=False, dropout_rate=0.0, use_crf=False, evaluate=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("data.pkl", "rb") as f:
        data = pickle.load(f)
    train, dev, test = data['train'], data['dev'], data['test']
    word2id, tag2id, label2id, embedding = data['w2id'], data['t2id'], data['l2id'], data['embedding']

    id2label = {i: l for l, i in label2id.items()}
    model = Model((word2id, tag2id, label2id),
                  word_dim=word_dim, pos_dim=pos_dim, hidden_size=hidden_size, rnn_layers=rnn_layers,
                  dropout_rate=dropout_rate, bidirectional=bidirectional,
                  embedding=embedding if use_pretrained else None, use_crf=use_crf)

    model.to(device)

    parameters = [p for p in model.parameters() if p.requires_grad]

    optimizer = optim.Adam(parameters)

    if evaluate:
        state_dict = torch.load(os.path.join("save", "best_model.pt"))
        model.load_state_dict(state_dict)
        print("Load Best Model in %s" % os.path.join("save", "best_model.pt"))

        model_eval(model, dev, test, device, id2label)
        exit()

    best_F1 = 0.0
    losses = []
    for epoch in range(1, epochs + 1):
        print("Epoch %3d....." % epoch)
        num_data = len(train)
        num_batch = (num_data + batch_size - 1) // batch_size

        model.train()
        print("Start Training in Epoch %3d" % epoch)
        for ii in range(num_batch):

            batch_word_ids, batch_tag_ids, batch_labels_ids = batchify(ii, batch_size, num_data, train)

            batch_word_ids = batch_word_ids.to(device)
            batch_tag_ids = batch_tag_ids.to(device)
            batch_labels_ids = batch_labels_ids.to(device)

            loss = model(batch_word_ids, batch_tag_ids, batch_labels_ids)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            losses.append(loss.data)

            if (ii + 1) % 100 == 0:
                print("%6d/%6d: loss %.6f" % (ii + 1, num_batch, sum(losses) / len(losses)))
                losses = []

        # Dev Evaluation
        num_data = len(dev)
        num_batch = (num_data + batch_size - 1) // batch_size

        model.eval()
        print("Dev Evaluation in Epoch %3d" % epoch)

        total_answer_ids, total_pred_ids = [], []
        for ii in range(num_batch):
            batch_word_ids, batch_tag_ids, batch_labels_ids = batchify(ii, batch_size, num_data, dev)

            batch_word_ids = batch_word_ids.to(device)
            batch_tag_ids = batch_tag_ids.to(device)
            batch_labels_ids = batch_labels_ids.to(device)

            batch_answer_ids, batch_pred_ids = model(batch_word_ids, batch_tag_ids, batch_labels_ids)

            total_answer_ids.extend(batch_answer_ids)
            total_pred_ids.extend(batch_pred_ids)

        precision, recall, F1 = evaluate_ner_F1(total_answer_ids, total_pred_ids, id2label)

        print("[Epoch %d][ Dev] precision : %.2f, recall : %.2f, F1 : %.2f" % (epoch, precision, recall, F1))

        if F1 > best_F1:
            torch.save(model.state_dict(), os.path.join("save", "best_model.pt"))
            best_F1 = F1
            print('[new best model saved.]')


def model_eval(model, dev, test, device, id2label):
    # Dev Evaluation
    num_data = len(dev)
    num_batch = (num_data + batch_size - 1) // batch_size

    model.eval()
    print("Dev Evaluation in Best Model")

    total_answer_ids, total_pred_ids = [], []
    total_words = [sent[3] for sent in dev]
    for ii in range(num_batch):
        batch_word_ids, batch_tag_ids, batch_labels_ids = batchify(ii, batch_size, num_data, dev)

        batch_word_ids = batch_word_ids.to(device)
        batch_tag_ids = batch_tag_ids.to(device)
        batch_labels_ids = batch_labels_ids.to(device)

        batch_answer_ids, batch_pred_ids = model(batch_word_ids, batch_tag_ids, batch_labels_ids)

        total_answer_ids.extend(batch_answer_ids)
        total_pred_ids.extend(batch_pred_ids)

    precision, recall, F1 = evaluate_ner_F1_and_write_result(total_words, total_answer_ids, total_pred_ids, id2label, setname='dev')

    print("[Best][ Dev] precision : %.2f, recall : %.2f, F1 : %.2f" % (precision, recall, F1))

    # Test Evaluation
    num_data = len(test)
    num_batch = (num_data + batch_size - 1) // batch_size

    model.eval()
    print("Test Evaluation in Best Model")

    total_answer_ids, total_pred_ids = [], []
    total_words = [sent[3] for sent in test]
    for ii in range(num_batch):
        batch_word_ids, batch_tag_ids, batch_labels_ids = batchify(ii, batch_size, num_data, test)

        batch_word_ids = batch_word_ids.to(device)
        batch_tag_ids = batch_tag_ids.to(device)
        batch_labels_ids = batch_labels_ids.to(device)

        batch_answer_ids, batch_pred_ids = model(batch_word_ids, batch_tag_ids, batch_labels_ids)

        total_answer_ids.extend(batch_answer_ids)
        total_pred_ids.extend(batch_pred_ids)

    precision, recall, F1 = evaluate_ner_F1_and_write_result(total_words, total_answer_ids, total_pred_ids, id2label, setname='test')

    print("[Best][Test] precision : %.2f, recall : %.2f, F1 : %.2f" % (precision, recall, F1))


if __name__ == "__main__":
    batch_size = 64
    epochs = 60
    word_dim = 100
    pos_dim = 50
    hidden_size = 256
    rnn_layers = 2
    dropout_rate = 0.33
    bidirectional = True
    use_pretrained = True
    use_crf = False
    evaluate = True

    train(epochs=epochs,
          batch_size=batch_size,
          word_dim=word_dim,
          pos_dim=pos_dim,
          hidden_size=hidden_size,
          rnn_layers=rnn_layers,
          bidirectional=bidirectional,
          use_pretrained=use_pretrained,
          use_crf=use_crf,
          dropout_rate=dropout_rate,
          evaluate=evaluate)

    train(epochs=epochs,
          batch_size=batch_size,
          word_dim=word_dim,
          pos_dim=pos_dim,
          hidden_size=hidden_size,
          rnn_layers=rnn_layers,
          bidirectional=bidirectional,
          use_pretrained=use_pretrained,
          use_crf=use_crf,
          dropout_rate=dropout_rate,
          evaluate=False)
