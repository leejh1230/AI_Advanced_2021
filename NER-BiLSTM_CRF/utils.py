import torch
from torch.nn.utils.rnn import pad_sequence
import os


def get_chunk_type(tag_name):
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type


def get_chunks(seq):
    default = "O"

    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass

    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks


def batchify(ii, batch_size, num_data, data):
    start = ii * batch_size
    end = num_data if (ii + 1) * batch_size > num_data else (ii + 1) * batch_size

    batch_data = data[start:end]

    batch_word_ids = [torch.tensor(data[0], dtype=torch.long) for data in batch_data]
    batch_tag_ids = [torch.tensor(data[1], dtype=torch.long) for data in batch_data]
    batch_labels_ids = [torch.tensor(data[2], dtype=torch.long) for data in batch_data]

    batch_word_ids = pad_sequence(batch_word_ids, batch_first=True)
    batch_tag_ids = pad_sequence(batch_tag_ids, batch_first=True)
    batch_labels_ids = pad_sequence(batch_labels_ids, batch_first=True)

    return batch_word_ids, batch_tag_ids, batch_labels_ids


def evaluate_ner_F1(total_answer_ids, total_pred_ids, id2label):
    num_match = num_preds = num_answers = 0

    for answer_ids, pred_ids in zip(total_answer_ids, total_pred_ids):
        answers = [id2label[l_id] for l_id in answer_ids]
        preds = [id2label[l_id] for l_id in pred_ids]

        answer_seg_result = set(get_chunks(answers))
        pred_seg_result = set(get_chunks(preds))

        num_match += len(answer_seg_result & pred_seg_result)
        num_answers += len(answer_seg_result)
        num_preds += len(pred_seg_result)

    precision = 100.0 * num_match / num_preds
    recall = 100.0 * num_match / num_answers
    F1 = 2 * precision * recall / (precision + recall)

    return precision, recall, F1


def evaluate_ner_F1_and_write_result(total_words, total_answer_ids, total_pred_ids, id2label, setname):
    num_match = num_preds = num_answers = 0

    filename = os.path.join("save", "best_%s_result.txt" % setname)
    of = open(filename, "w")
    for words, answer_ids, pred_ids in zip(total_words, total_answer_ids, total_pred_ids):
        answers = [id2label[l_id] for l_id in answer_ids]
        preds = [id2label[l_id] for l_id in pred_ids]

        answer_seg_result = set(get_chunks(answers))
        pred_seg_result = set(get_chunks(preds))

        num_match += len(answer_seg_result & pred_seg_result)
        num_answers += len(answer_seg_result)
        num_preds += len(pred_seg_result)

        for w, a_l, p_l in zip(words, answers, preds):
            of.write("\t".join([w, a_l, p_l]) + "\n")
        of.write("\n")
    of.close()

    precision = 100.0 * num_match / num_preds
    recall = 100.0 * num_match / num_answers
    F1 = 2 * precision * recall / (precision + recall)

    return precision, recall, F1
