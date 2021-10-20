import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from torch.utils import data

model_path = 'train_model/best_model'
data_path = 'data/FB15K'
train_batch_size = 1024
eval_batch_size = 512
epochs = 1000
learning_rate = 0.1

hidden_size = 50
eval_freq = 25
margin = 1.
seed = 3435

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_dict(path) :
    output_dict = {}

    for line in open(path, 'r').readlines()[1:] :
        key, value = line.strip().split('\t')
        output_dict[key] = int(value)

    return output_dict

class FB15K_Dataset(data.Dataset) :
    def __init__(self, path, entity2id, relation2id):
        self.entity2id = entity2id
        self.relation2id = relation2id

        self.data = []
        for line in open(path, 'r').readlines()[1:]:
            self.data.append(list(map(int, line.strip().split())))
            # str -> int (sbj_id, obj_id, rel_id)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

class TransE(nn.Module):
    def __init__(self, n_entity, n_relation, hidden_size, margin=1.0, device=True):
        super(TransE, self).__init__()
        self.device = device

        self.n_entity = n_entity
        self.n_relation = n_relation
        self.hidden_size = hidden_size

        self.entity_embedding = nn.Embedding(self.n_entity + 1, self.hidden_size, padding_idx=self.n_entity)
        self.relation_embedding = nn.Embedding(self.n_relation + 1, self.hidden_size, padding_idx=self.n_relation)

        self.init_weight(self.entity_embedding)
        self.init_weight(self.relation_embedding)

        self.loss_func = nn.MarginRankingLoss(margin=margin, reduction='none')

    def init_weight(self, embedding):
        n_vocab, hidden_dim = embedding.weight.data.size()
        sqrt_dim = hidden_dim ** 0.5

        embedding.weight.data = torch.FloatTensor(n_vocab, hidden_dim).uniform_(-6./sqrt_dim, 6./sqrt_dim)
        embedding.weight.data = F.normalize(embedding.weight.data, 2, 1)

    def get_score(self, triple):
        sbj, rel, obj = triple[:, 0], triple[:, 1], triple[:, 2]

        sbj_embedding = self.entity_embedding(sbj)
        rel_embedding = self.relation_embedding(rel)
        obj_embedding = self.entity_embedding(obj)

        score = torch.norm((sbj_embedding + rel_embedding - obj_embedding), p=1, dim=1)

        return score

    def forward(self, positive_triple, negative_triple):
        positive_score = self.get_score(positive_triple)
        negative_score = self.get_score(negative_triple)

        y = torch.tensor([-1.], dtype=torch.float, device=self.device)

        return self.loss_func(positive_score, negative_score, y)

def hit_at_k(pred, answer, device, k=10) :
    zero_tensor = torch.tensor([0], device=device)
    one_tensor = torch.tensor([1], device=device)

    _, indices = pred.topk(k=k, largest=False)

    return torch.where(indices == answer.unsqueeze(1), one_tensor, zero_tensor).sum().item()

def MRR(pred, answer) :
    return (1. / (pred.argsort() == answer.unsqueeze(1)).nonzero()[:, 1].float().add(1.)).sum().item()

def evaluation(model, data_loader, device) :
    model.eval() # evaluation mode
    hit_at_1, hit_at_3, hit_at_10, mrr, total = 0., 0., 0., 0., 0.

    entity_ids = torch.arange(model.n_entity, device=device).unsqueeze(0)

    pbar = tqdm(data_loader, total=len(data_loader), desc='evaluation...')
    for sbj, obj, rel in pbar :
        sbj, rel, obj = sbj.to(device), rel.to(device), obj.to(device)  # to GPU
        b_size = sbj.size(0)

        all_entity = entity_ids.repeat(b_size, 1)
        repeat_sbj = sbj.unsqueeze(1).repeat(1, all_entity.size(1))
        repeat_rel = rel.unsqueeze(1).repeat(1, all_entity.size(1))
        repeat_obj = obj.unsqueeze(1).repeat(1, all_entity.size(1))

        sbj_triples = torch.stack((repeat_sbj, repeat_rel, all_entity), dim=2).view(-1, 3)
        obj_triples = torch.stack((all_entity, repeat_rel, repeat_obj), dim=2).view(-1, 3)

        obj_pred_score = model.get_score(sbj_triples).view(b_size, -1)
        sbj_pred_score = model.get_score(obj_triples).view(b_size, -1)

        pred = torch.cat([sbj_pred_score, obj_pred_score], dim=0)
        answer = torch.cat([sbj, obj], dim=0)

        hit_at_1 += hit_at_k(pred, answer, device, k=1)
        hit_at_3 += hit_at_k(pred, answer, device, k=3)
        hit_at_10 += hit_at_k(pred, answer, device, k=10)

        mrr += MRR(pred, answer)
        total += pred.size(0)

    hit_at_1_score = hit_at_1 / total * 100.
    hit_at_3_score = hit_at_3 / total * 100.
    hit_at_10_score = hit_at_10 / total * 100.
    mrr_score = mrr / total * 100.

    return hit_at_1_score, hit_at_3_score, hit_at_10_score, mrr_score

if __name__ == '__main__' :
    # 1. Load data set
    entity2id = load_dict(os.path.join(data_path, 'entity2id.txt'))
    relation2id = load_dict(os.path.join(data_path, 'relation2id.txt'))

    train_dataset = FB15K_Dataset(os.path.join(data_path, 'train2id.txt'), entity2id, relation2id)
    dev_dataset = FB15K_Dataset(os.path.join(data_path, 'valid2id.txt'), entity2id, relation2id)
    test_dataset = FB15K_Dataset(os.path.join(data_path, 'test2id.txt'), entity2id, relation2id)

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=train_batch_size,
                                   shuffle=True)
    dev_loader = data.DataLoader(dev_dataset,
                                 batch_size=eval_batch_size,
                                 shuffle=False)
    test_loader = data.DataLoader(test_dataset,
                                  batch_size=eval_batch_size,
                                  shuffle=False)
    # 2. Model
    model = TransE(n_entity=len(entity2id),
                   n_relation=len(relation2id),
                   hidden_size=hidden_size,
                   margin=margin,
                   device=device)
    model.to(device) # to GPU
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    print('Model Structure : {}'.format(model))

    # 3. Training
    pbar = tqdm(range(1, epochs+1), total=epochs)
    best_score = 0.
    for epoch in pbar :
        model.train() # train mode
        for i, (sbj, obj, rel) in enumerate(train_loader) :
            sbj, rel, obj = sbj.to(device), rel.to(device), obj.to(device)  # to GPU

            positive_triples = torch.stack((sbj, rel, obj), dim=1) # (batch) * 3 -> (batch, 3)

            # Negative sampling
            head_or_tail = torch.randint(high=2, size=sbj.size(), device=device)
            random_entities = torch.randint(high=len(entity2id), size=sbj.size(), device=device)
            neg_sbj = torch.where(head_or_tail == 1, random_entities, sbj)
            neg_obj = torch.where(head_or_tail == 0, random_entities, obj)
            negative_triples = torch.stack((neg_sbj, rel, neg_obj), dim=1) # (batch) * 3 -> (batch, 3)

            loss = model(positive_triples, negative_triples).mean()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        pbar.set_description('Epoch = {}, loss = {:.6f}'. format(epoch, loss))
        if(epoch % eval_freq == 0) :
            hit_at_1_score, hit_at_3_score, hit_at_10_score, mrr_score = evaluation(model, dev_loader, device)

            print('hit@1 : {:.2f}, hit@3 : {:.2f}, hit@10 : {:.2f}, mrr : {:.2f}'.format(hit_at_1_score,
                                                                                         hit_at_3_score,
                                                                                         hit_at_10_score,
                                                                                         mrr_score))
            if(hit_at_10_score > best_score) :
                print('best model save...')
                state_dict = model.state_dict()
                torch.save(state_dict, model_path)

    model.load_state_dict(torch.load(model_path))
    hit_at_1_score, hit_at_3_score, hit_at_10_score, mrr_score = evaluation(model, test_loader, device)