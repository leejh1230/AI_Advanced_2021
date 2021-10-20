import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torch.optim as optim
import pickle
from transformers import AutoModel
from transformers.optimization import AdamW, get_linear_schedule_with_warmup


class Model(nn.Module):
    def __init__(self, bert_dim, dropout_rate):
        super(Model, self).__init__()

        self.bert = AutoModel.from_pretrained('bert-base-multilingual-cased')
        self.linear = nn.Linear(bert_dim, 1)

        self.dropout_rate = dropout_rate

    def forward(self, bert_ids):
        bert_seq_outputs, bert_output = self.bert(bert_ids)

        output = torch.sigmoid(self.linear(bert_output)).squeeze(1)

        return output


def train():
    batch_size = 16
    epochs = 5
    gradient_accumulation_steps = 1
    bert_lr = 1.5e-5
    adam_epsilon = 1e-8
    warmup_steps = 4000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("data.pkl", "rb") as f:
        data = pickle.load(f)
    train, test = data['train'], data['test']

    model = Model(bert_dim=768, dropout_rate=0.3)
    model.to(device)

    criterion = nn.BCELoss()

    # Building optimizer.
    bert_parameters = [p for p in model.named_parameters() if p[1].requires_grad]
    bert_parameters = [n for n in bert_parameters if 'pooler' not in n[0]]

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_parameters if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in bert_parameters if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}, ]

    train_size = (len(train) + batch_size - 1) // batch_size

    t_total = int(train_size / gradient_accumulation_steps * epochs)

    optimizer = AdamW(optimizer_grouped_parameters, lr=bert_lr, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

    losses = []
    for epoch in range(1, epochs + 1):
        num_data = len(train)
        num_batch = (num_data + batch_size - 1) // batch_size

        model.train()
        for ii in range(num_batch):
            start = ii * batch_size
            end = num_data if (ii + 1) * batch_size > num_data else (ii + 1) * batch_size

            batch_data = train[start:end]

            batch_bert_ids = [torch.tensor(data[0], dtype=torch.long) for data in batch_data]
            batch_labels_ids = [data[1] for data in batch_data]

            batch_bert_ids = pad_sequence(batch_bert_ids, batch_first=True)

            batch_labels_ids = torch.tensor(batch_labels_ids, dtype=torch.float)

            batch_bert_ids = batch_bert_ids.to(device)
            batch_labels_ids = batch_labels_ids.to(device)

            batch_outputs = model(batch_bert_ids)

            loss = criterion(batch_outputs, batch_labels_ids)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            scheduler.step()

            losses.append(loss.data)

            if (ii + 1) % 200 == 0:
                print("%6d/%6d: loss %.6f" % (ii + 1, num_batch, sum(losses) / len(losses)))
                losses = []

        num_data = len(test)
        num_batch = (num_data + batch_size - 1) // batch_size

        model.eval()

        total = len(test)
        match = 0
        for ii in range(num_batch):
            start = ii * batch_size
            end = num_data if (ii + 1) * batch_size > num_data else (ii + 1) * batch_size

            batch_data = test[start:end]

            batch_bert_ids = [torch.tensor(data[0], dtype=torch.long) for data in batch_data]
            batch_labels_ids = [data[1] for data in batch_data]

            batch_bert_ids = pad_sequence(batch_bert_ids, batch_first=True)

            batch_bert_ids = batch_bert_ids.to(device)

            batch_outputs = model(batch_bert_ids)

            batch_outputs = batch_outputs.data.cpu().numpy().tolist()

            batch_pred_ids = [1 if output >= 0.5 else 0 for output in batch_outputs]

            for a, o in zip(batch_labels_ids, batch_pred_ids):
                if a == o:
                    match += 1

        print("Epoch %d, match : %6d, total : %6d, ACC : %.2f" % (epoch, match, total, 100 * match / total))


if __name__ == "__main__":
    train()
