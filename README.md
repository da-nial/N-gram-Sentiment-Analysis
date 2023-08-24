## N-gram-Sentiment-Analysis

This project in trains n-gram models from scratch for sentence polarity detection.
The goal is to determine whether an input sentence is positive or negative.

The complete project description can be found in [instructions.pdf](docs/instructions.pdf)
and [report.pdf](docs/report.pdf).

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
    3. Probability of a sentence belonging to each class: $$P(l_i|w_1 w_2 w_3 \dots w_n) \propto P(w_1 w_2 \dots
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

| λ_1 | λ_2 | λ_3 | ε   | cutoff? | Unigram  |       | Bigram   |       |
|-----|-----|-----|-----|---------|----------|-------|----------|-------|
|     |     |     |     |         | Accuracy | F1    | Accuracy | F1    |
| 0.1 | 0.3 | 0.6 | 0.5 | no      | 0.600    | 0.402 | 0.667    | 0.677 |
| 0.1 | 0.3 | 0.6 | 0.5 | yes     | 0.508    | 0.033 | 0.604    | 0.492 |
| 0.2 | 0.3 | 0.5 | 0.3 | no      | 0.606    | 0.423 | 0.663    | 0.664 |
| 0.2 | 0.3 | 0.5 | 0.3 | yes     | 0.501    | 0.022 | 0.600    | 0.492 |
| 0.2 | 0.2 | 0.6 | 0.3 | yes     | 0.606    | 0.444 | 0.644    | 0.652 |
| 0.1 | 0.2 | 0.7 | 0.3 | yes     | 0.602    | 0.408 | 0.631    | 0.640 |
| 0.1 | 0.1 | 0.8 | 0.3 | yes     | 0.616    | 0.445 | 0.659    | 0.675 |

The cutoff column indicates whether least and most common words were discarded or not.

### Run:

```python main.py```

Sample output:

```
Using bigram_predict
#True Positives: 355, #False Positives: 180
#True Negatives: 354, #False Negatives: 179
#Accuracy: 0.6638576779026217
Precision: 0.6647940074906367, Recall: 0.6635514018691588
F1 Score: 0.6641721234798877

Enter your sentence...
```

### Course Information

- **Course**: Artificial Intelligence
- **University**: Amirkabir University of Technology
- **Semester**: Spring 2021

Let me know if you have any questions.
