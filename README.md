# Tweet-level Arabic Dialect Identification

## Mohammad Abdeh

### January 2021

### Abstract

Arabic Dialect Identification (ADI) on written text is the task of classifying the dialect of a given Arabic text. ADI is a challenging, yet a crucial task, having a significant role in machine translation and speaker identification. 
In this work, we consider the task of tweet-level ADI using the recently proposed QADI dataset. We compare the results of training and testing the following standalone models:
- The base ArabicBERT, a pretrained BERT-base language model for Arabic.
- The medium ArabicBERT, a pretrained BERT-medium language model for Arabic.
- A feature engineering approach, using word-level and character-level n-grams to train a multi-layer perceptron (MLP).

We also present a novel pipeline that uses these models and applies various ensemble methods. The medium ArabicBERT model showed the best results when trained for 8 epochs among the standalone models with an F1-score of 77.03. Our Weighted Soft Voting approach has shown the best results with a macro-averaged F1-score of 78.80% across 18 classes, achieving a new state-of-the-art result in tweet-level multi-class ADI.

### 1. Introduction

The Arabic Language is one of the oldest and richest languages, spoken by over 400 million people worldwide. It is the 4th most used language on the Internet. Arabic has three variations:
- **Classical Arabic (CA):** Used in the Quran, poetry, and old scriptures.
- **Modern Standard Arabic (MSA):** The formal variant used in education, print, and law.
- **Dialectical Arabic (DA):** Used in daily life and informal communications.

Most Arabic Natural Language Processing (NLP) research has focused on MSA, but the rise of DA material on social media has increased interest in developing datasets and models to process DA. 
This work focuses on the tweet-level classification of 18 Arabic dialects using the QADI dataset. The main contributions are:
1. Fine-tuning existing transformer-based pre-trained models (AraBERT, medium ArabicBERT, and base ArabicBERT) on the QADI dataset.
2. Develope a feature engineering approach that uses word-level and character-level n-grams.
3. Evaluate and compare the results of the models.
4. Developing a pipeline that sets new state-of-the-art results by applying different ensembling methods on the results of the previously mentioned standalone models.

### 2. Related Work

The interest in this task has significantly increased in the last decade. Early works used language modeling approaches, and many shared tasks were organized to tackle ADI. For example:
- The Discriminating between Similar Languages (DSL) shared task at VarDial2016 focused on identifying Arabic dialects in speech transcripts.
- The MADAR shared task included sub-tasks like identifying city-level and country-level dialects from tweets.

BERT-based models, especially AraBERT, have become state-of-the-art for ADI tasks.

### 3. Proposed Approach

In this work, ArabicBERT models (medium and base) were preferred over AraBERT due to their larger pre-training datasets and inclusion of dialectal data. The proposed pipeline involves the following steps:

#### 3.1 Preprocessing

- User mentions and URLs were replaced with placeholders.
- Emojis and hashtag signs were removed.
- Farasa segmenter was used for sub-word unit segmentation, for instance, the word "wKalimatuna" "كلمتنا" meaning "and our word" can be partitioned into the prex "w" "و" and the stem "kalimatu" "كلمة"and the suffixx "na" "نا".
After the normalization and segmentation processes, we started building up thefollowing classiers.


#### 3.2 ArabicBERT

- Fine-tuned the medium and base ArabicBERT models on the QADI dataset by adding a dense layer and a softmax classifier on top of the pre-trained model.
- We have trained the medium, base for 8, and 4 ephocs respectively with maximum sequence length of 128 using the Adam optimizer to minimize the cross-entropy with a decaying learning rate and a batch size of 16.


#### 3.3 N-grams Model

- Extracted and vectorized word-level (unigrams and bigrams) and character-level (2-6 sub-words) n-grams using TF-IDF scores.
- Trained an Multi-layer perceptron (MLP) with one hidden layer consisting of 64 hidden units.
-  The mlp has been trained using the extracted features for 5 ephocs, a batch size of 32 and a learning rate of 0.0001.

#### 3.4 Ensembling Methods

- **Hard Majority Voting**: Selected the prediction most predicted by the candidate models.
- **Unweighted Soft Voting**: Averaged softmax probabilities across the candidate models.
- **Weighted Soft Voting**: Assigned different weights for each model and averaged softmax probabilities.


### 4. Experiment Settings

#### 4.1 Dataset

The QADI dataset consists of 540k training tweets and 3502 test tweets, covering 18 dialects.

#### 4.2 Baseline

The AraBERT model was fine-tuned on the QADI dataset as a baseline.

### 5. Experiments

Reported results of evaluating the models on the QADI test set. The medium ArabicBERT trained for 8 epochs showed the best standalone performance. Ensemble methods, especially weighted soft voting, improved the results further.

![Experiment Results](path/to/experiment_results_image.png)

### 6. Discussion

Discussed classification errors, mainly due to the misclassification of geographically close dialects, and provided an error analysis.

![Error Analysis](path/to/error_analysis_image.png)

### 7. Conclusion

Outlined the contributions of the work and summarized the results.

![Conclusion](path/to/conclusion_image.png)

### References

1. Ali, Ahmed, et al. "The MGB-3 Arabic dialect speech recognition challenge." 2017 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU). IEEE, 2017.
2. Ali, Ahmed, et al. "A complete KALDI recipe for building Arabic speech recognition systems." Proceedings of the 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2017.
3. El-Haj, Mahmoud, Reem Suwaileh, and Paul Rayson. "Multidialect Arabic Corpus for Text Classification." Proceedings of the 11th International Conference on Language Resources and Evaluation (LREC 2018). 2018.
4. Safaya, Ali, Moutasem Abdullatif, and Deniz Yuret. "KUISAIL at SemEval-2020 task 12: BERT-CNN for offensive speech identification in social media." Proceedings of the Fourteenth Workshop on Semantic Evaluation. 2020.

