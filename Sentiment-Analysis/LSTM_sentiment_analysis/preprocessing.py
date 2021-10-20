import os
import operator
import pickle


# 데이터를 읽어옴
def read_data(filename):
    dataset = []
    sent = []
    for line in open(filename):
        line = line.strip()

        if not line:
            doc_id, label = sent[-1].split("\t")

            data = ([item.split(" ") for item in sent[:-1]], int(label))
            dataset.append(data)
            sent = []
            continue

        sent.append(line)

    return dataset


# vocab 생성
def make_vocab(dataset):
    words = {}
    tags = {}

    for data in dataset:
        sent, label = data[0], data[1]

        for m, t in sent:
            if m not in words:
                words[m] = 0
            words[m] += 1

            if t not in tags:
                tags[t] = 0
            tags[t] += 1

    sorted_words = sorted(words.items(), key=operator.itemgetter(1), reverse=True)
    sorted_tags = sorted(tags.items(), key=operator.itemgetter(1), reverse=True)

    word2id = {w: i + 2 for i, (w, c) in enumerate(sorted_words)}
    tag2id = {w: i for i, (w, c) in enumerate(sorted_tags)}

    word2id['<PAD>'] = 0
    word2id['<UNK>'] = 1

    return word2id, tag2id


# word 를 index 로 변경
def convert_ids(word2id, tag2id, data, UNK=1):
    sent, label = data[0], data[1]

    word_ids = [word2id[w] if w in word2id else UNK for w, t in sent]
    tag_ids = [tag2id[t] if t in tag2id else UNK for w, t in sent]

    return word_ids, tag_ids, label


def main():
    traindata = read_data(os.path.join("data", "train_tagged.txt"))
    testdata = read_data(os.path.join("data", "test_tagged.txt"))

    word2id, tag2id = make_vocab(traindata)

    train = [convert_ids(word2id, tag2id, data) for data in traindata]
    test = [convert_ids(word2id, tag2id, data) for data in testdata]

    data = {'train': train, 'test': test, 'w2id': word2id, 't2id': tag2id}

    with open('data.pkl', 'wb') as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    main()
