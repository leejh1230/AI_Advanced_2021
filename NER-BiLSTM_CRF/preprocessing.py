import os
import operator
import pickle
import numpy as np


def read_data(filename):
    sents = []
    sent = []
    for line in open(filename):
        line = line.strip()

        if not line:
            sents.append(sent)
            sent = []
            continue

        word, tag, chunk, label = line.split(' ')
        word = word.lower()

        sent.append((word, tag, chunk, label))

    return sents


def make_vocab(train):
    word_dict = {}
    tag_dict = {}
    label_dict = {}
    for sent in train:
        # word, tag, chunk, label
        words = [item[0] for item in sent]
        tags = [item[1] for item in sent]
        labels = [item[3] for item in sent]

        for w, t, l in zip(words, tags, labels):
            if w not in word_dict:
                word_dict[w] = 0
            word_dict[w] += 1

            if t not in tag_dict:
                tag_dict[t] = 0
            tag_dict[t] += 1

            if l not in label_dict:
                label_dict[l] = 0
            label_dict[l] += 1

    sorted_words = sorted(word_dict.items(), key=operator.itemgetter(1), reverse=True)
    sorted_tags = sorted(tag_dict.items(), key=operator.itemgetter(1), reverse=True)
    sorted_labels = sorted(label_dict.items(), key=operator.itemgetter(1), reverse=True)

    word2id = {w: i + 2 for i, (w, c) in enumerate(sorted_words)}
    tag2id = {w: i + 2 for i, (w, c) in enumerate(sorted_tags)}
    label2id = {w: i for i, (w, c) in enumerate(sorted_labels)}

    word2id['<PAD>'] = 0
    word2id['<UNK>'] = 1

    tag2id['<PAD>'] = 0
    tag2id['<UNK>'] = 1

    return word2id, tag2id, label2id


def convert_ids(word2id, tag2id, label2id, sent, UNK=1):
    # word, tag, chunk, label
    words = [item[0] for item in sent]
    tags = [item[1] for item in sent]
    labels = [item[3] for item in sent]

    word_ids = [word2id[w] if w in word2id else UNK for w in words]
    tag_ids = [tag2id[t] if t in tag2id else UNK for t in tags]
    label_ids = [label2id[l] for l in labels]

    return word_ids, tag_ids, label_ids, words


def build_pretrained_lookup_table(word2id, pretrained_emb_file, embedding_dim=100):
    embedding_dict = {}
    for line in open(pretrained_emb_file):
        tokens = line.split(' ')
        word, vecs = tokens[0], np.asarray(tokens[1:], np.float32)
        embedding_dict[word] = vecs

    scale = np.sqrt(3.0 / embedding_dim)

    vocab_size = len(word2id)

    embedding = np.random.uniform(-scale, scale, [vocab_size, embedding_dim]).astype(np.float32)

    for word, index in word2id.items():
        if word in embedding_dict:
            vec = embedding_dict[word]
            embedding[index, :] = vec

    return embedding


if __name__ == "__main__":
    traindata = read_data(os.path.join("data", "train.txt"))
    devdata = read_data(os.path.join("data", "valid.txt"))
    testdata = read_data(os.path.join("data", "test.txt"))

    word2id, tag2id, label2id = make_vocab(traindata)

    train = [convert_ids(word2id, tag2id, label2id, sent) for sent in traindata]
    dev = [convert_ids(word2id, tag2id, label2id, sent) for sent in devdata]
    test = [convert_ids(word2id, tag2id, label2id, sent) for sent in testdata]

    embedding = build_pretrained_lookup_table(word2id, os.path.join("data", "glove.6B.100d.txt"))

    data = {'train': train, 'dev': dev, 'test': test, 'w2id': word2id, 't2id': tag2id, 'l2id': label2id, 'embedding': embedding}

    with open('data.pkl', 'wb') as f:
        pickle.dump(data, f)
