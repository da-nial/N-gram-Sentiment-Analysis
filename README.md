## N-gram-Sentiment-Analysis README

N-gram-Sentiment-Analysis is a project in which n-gram models are trained from scratch for sentence polarity detection.
The goal is to determine whether an input sentence is positive or negative.

### Dataset

The dataset is divided into two files: `rt-polarity_neg.txt` containing negative sentences and `rt-polarity_pos.txt`
containing positive sentences. The dataset is split into train/test with a 90/10 ratio.

### Implementation

1. **Preprocessing**: Unimportant punctuations are removed and all letters are converted to lowercase to ensure a
   consistent format.

2. **Model Building**
    1. A dictionary of words from the dataset is created and the occurrences of each word in each class (positive and
       negative) are counted.
    2. Words that occur less than a minimum threshold and more than a maximum threshold are discarded to avoid taking
       special or general words into account.
    3. A dictionary of word pairs and their occurrences is created for the bigram model.

3. **Probability Calculation**
    1. Probability of each word in a class (positive or negative): $$P(W_i) = \frac{\text{count}(w_i)}{M}$$
    2. Probability of each word pair in a class: $$P(w_i|w_{i-1}) = \frac{\text{count}(w_{i-1} w_i)}{\text{count}(w_
       {i-1})}$$
    3. Probability of a sentence belonging to each class: $$P(l_i|w_1 w_2 w_3 \dots w_n) \approx P(w_1 w_2 \dots
       w_n|l_i) *
       P(l_i)$$
       The prior probability of a sentence belonging to positive or negative classes ($P(l_i)$) is assumed to be 0.5.
    4. To calculate the probability of a sentence belonging to a class: $$P(w_1 w_2 \dots w_n) = P(w_1) * \prod_
       {i=2}^{n} P(
       w_i|w_{i-1})$$
    5. To calculate $P(w_i|w_{i-1})$: $$P(w_i|w_{i-1}) = \lambda_3 * P(w_i|w_{i-1}) + \lambda_2 * P(w_i) + \lambda_1 *
       \epsilon$$, where $\lambda_1 + \lambda_2 + \lambda_3 = 1$ and $0 < \epsilon < 1$.

4. **Classification**: The class with the highest probability for the input sentence is selected.

### Results

| $$\lambda_1$$ | $$\lambda_2$$ | $$\lambda_3$$ | $$\epsilon$$ | Unigram | Bigram |
|---------------|---------------|---------------|--------------|---------|--------|
|               |               |               |              |         |        |

* Unigram and Bigram headers have two subheaders: Accuracy and F1
