def bigram_sentence_prob(sentence, words_prob, pairs_prob, l3, l2, l1, e):
    words = sentence.split()
    pairs = zip(words, words[1:])

    prob = words_prob.get(words[0], 0)

    for pair in pairs:
        word1, word2 = pair

        cur_pair_prob = pairs_prob.get(pair, 0)
        cur_pair_prob = (l3 * cur_pair_prob) + (l2 * words_prob.get(word1, 0)) + (l1 * e)

        prob *= cur_pair_prob

    return prob


def bigram_predict(sentence, pos_words_prob, pos_pairs_prob, neg_words_prob, neg_pairs_prob, tuning_params):
    if sentence == "":
        return "pos"

    positivity_prob = bigram_sentence_prob(sentence, pos_words_prob, pos_pairs_prob, **tuning_params)
    negativity_prob = bigram_sentence_prob(sentence, neg_words_prob, neg_pairs_prob, **tuning_params)

    return "pos" if positivity_prob > negativity_prob else "neg"
