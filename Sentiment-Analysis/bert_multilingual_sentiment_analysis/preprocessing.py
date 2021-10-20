import os
import operator
import pickle
from transformers import AutoModel, AutoTokenizer


def read_data(filename):
    datas = []
    f = open(filename)
    f.readline()
    for line in f:
        tokens = line.strip().split("\t")
        doc_id, sent, label = tokens[0], tokens[1], int(tokens[2])
        datas.append((doc_id, sent, label))

    return datas


def bert_convert_ids(data, tokenizer):
    doc_id, sent, label = data[0], data[1], data[2]

    tokens = ["[CLS]"] + tokenizer.tokenize(sent) + ["[SEP]"]

    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    return token_ids, label


if __name__ == "__main__":
    traindata = read_data(os.path.join("data", "train.txt"))
    testdata = read_data(os.path.join("data", "test.txt"))

    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

    train = [bert_convert_ids(data, tokenizer) for data in traindata]
    test = [bert_convert_ids(data, tokenizer) for data in testdata]

    data = {'train': train, 'test': test}

    with open('data.pkl', 'wb') as f:
        pickle.dump(data, f)
