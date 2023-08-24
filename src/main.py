from bigram import bigram_predict
from helper import read_ds, preprocess_ds, split_ds, get_probs, test, preprocess
from unigram import unigram_predict


def main():
    pos_comments_path = "./rt-polarity_pos.txt"
    neg_comments_path = "./rt-polarity_neg.txt"
    pos_ds, neg_ds = read_ds(pos_comments_path, neg_comments_path)

    pos_ds = preprocess_ds(pos_ds)
    neg_ds = preprocess_ds(neg_ds)

    split_rate = 0.9
    train_pos_ds, train_neg_ds, test_pos_ds, test_neg_ds = split_ds(pos_ds, neg_ds, split_rate)

    num_removing_commons = 10
    pos_words_prob, pos_pairs_prob = get_probs(train_pos_ds, num_removing_commons)
    neg_words_prob, neg_pairs_prob = get_probs(train_neg_ds, num_removing_commons)

    tuning_params = {
        "l3": 0.8,
        "l2": 0.1,
        "l1": 0.1,
        "e": 0.3
    }

    unigram_params = {"pos_words_prob": pos_words_prob,
                      "neg_words_prob": neg_words_prob}
    test(test_pos_ds, test_neg_ds, unigram_predict, unigram_params)

    bigram_params = {"pos_words_prob": pos_words_prob,
                     "pos_pairs_prob": pos_pairs_prob,
                     "neg_words_prob": neg_words_prob,
                     "neg_pairs_prob": neg_pairs_prob,
                     "tuning_params": tuning_params}
    test(test_pos_ds, test_neg_ds, bigram_predict, bigram_params)

    sentence = input()
    while sentence != "!q":
        sentence = preprocess(sentence)

        prediction = bigram_predict(sentence, pos_words_prob, pos_pairs_prob, neg_words_prob, neg_pairs_prob,
                                    tuning_params)

        print(prediction)

        sentence = input()


if __name__ == '__main__':
    main()
