from collections import Counter
from random import shuffle
from string import punctuation


def read_ds(pos_comments_path, neg_comments_path):
    with open(pos_comments_path, "r") as f:
        pos_ds = f.read().split("\n")

    with open(neg_comments_path, "r") as f:
        neg_ds = f.read().split("\n")

    return pos_ds, neg_ds


def split_ds(pos_ds, neg_ds, split_rate):
    shuffle(pos_ds)
    shuffle(neg_ds)

    pos_ds_bound = int(len(pos_ds) * split_rate)
    train_pos_ds, test_pos_ds = pos_ds[:pos_ds_bound], pos_ds[pos_ds_bound:]

    neg_ds_bound = int(len(neg_ds) * split_rate)
    train_neg_ds, test_neg_ds = neg_ds[:neg_ds_bound], neg_ds[neg_ds_bound:]

    return train_pos_ds, train_neg_ds, test_pos_ds, test_neg_ds


def preprocess_ds(ds):
    ds = [preprocess(comment) for comment in ds]
    return ds


def preprocess(sentence):
    sentence = remove_punctuation(sentence)
    sentence = sentence.lower()
    return sentence


def remove_punctuation(sentence):
    translation = str.maketrans('', '', punctuation)
    sentence = sentence.translate(translation)

    return sentence


def remove_stopwords(ds, word_counts, num_removing_commons):
    least_common = [word for word, count in word_counts.items() if count < 2]
    most_common = word_counts.most_common(num_removing_commons)

    for comment in ds:
        for word, count in most_common:
            comment.replace(word, "")

        for word in least_common:
            comment.replace(word, "")

    return ds


def count_words(ds, num_removing_commons):
    word_counts = Counter()
    for comment in ds:
        word_counts.update(word for word in comment.split())

    # word_counts = Counter({word: count for word, count in word_counts.items() if count >= 2})
    # most_common = word_counts.most_common(num_removing_commons)
    # for word, count in most_common:
    #     del word_counts[word]

    return word_counts


def count_pairs(ds):
    pairs_count = Counter()
    for comment in ds:
        words = comment.split()
        pairs = zip(words, words[1:])
        pairs_count.update(pair for pair in pairs)

    return pairs_count


def calc_words_prob(words_count):
    words_prob = {}
    m = sum(words_count.values())

    for word, count in words_count.items():
        prob = count / m
        words_prob.update({word: prob})

    return words_prob


def calc_pairs_prob(words_count, pairs_count):
    pairs_prob = {}

    for pair, count in pairs_count.items():
        word_1, word_2 = pair
        if word_1 in words_count:
            prob = count / words_count[word_1]
        else:
            prob = 0

        pairs_prob.update({pair: prob})

    return pairs_prob


def get_probs(ds, num_removing_commons=10):
    words_count = count_words(ds, num_removing_commons)
    pairs_count = count_pairs(ds)

    # remove_stopwords(ds, words_count, 10)

    words_prob = calc_words_prob(words_count)
    pairs_prob = calc_pairs_prob(words_count, pairs_count)

    return words_prob, pairs_prob


def test_ds(ds, reality, prediction_fn, prediction_params):
    hits = 0
    for comment in ds:
        prediction = prediction_fn(comment, **prediction_params)
        if prediction == reality:
            hits += 1

    misses = len(ds) - hits
    return hits, misses


def test(pos_ds, neg_ds, predict_fn, predict_params):
    true_positives, false_negatives = test_ds(pos_ds, "pos",
                                              predict_fn, predict_params)

    true_negatives, false_positives = test_ds(neg_ds, "neg",
                                              predict_fn, predict_params)

    m = len(pos_ds) + len(neg_ds)
    accuracy = (true_positives + true_negatives) / m
    precision = true_positives / (true_positives + false_negatives)
    recall = true_positives / (true_positives + false_positives)
    f1 = (2 * precision * recall) / (precision + recall)

    print(f"Using {predict_fn.__name__}")
    print(f"#True Positives: {true_positives}, #False Positives: {false_positives}")
    print(f"#True Negatives: {true_negatives}, #False Negatives: {false_negatives}")
    print(f"#Accuracy: {accuracy}")
    print(f"Precision: {precision}, Recall: {recall}")
    print(f"F1 Score: {f1}")
    print()
