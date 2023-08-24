def unigram_sentence_prob(sentence, words_prob):
    words = sentence.split()

    prob = 1
    for word in words:
        prob *= words_prob.get(word, 0)

    return prob


def unigram_predict(sentence, pos_words_prob, neg_words_prob):
    positivity_prob = unigram_sentence_prob(sentence, pos_words_prob)
    negativity_prob = unigram_sentence_prob(sentence, neg_words_prob)

    return "pos" if positivity_prob > negativity_prob else "neg"
