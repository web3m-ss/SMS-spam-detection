# SMS/Email Spam Detection using Machine Learning (NLP)

An end-to-end Natural Language Processing (NLP) pipeline for SMS spam classification. This project cleanses raw text, extracts features via TF-IDF, and combines multiple machine learning classifiers using a **Soft Voting Ensemble** to achieve high prediction accuracy (~99%).

---

##  Key Features

* **Advanced Text Preprocessing:** Regex-based cleaning for URLs, currency symbols (`$`, `£`), phone numbers, and arbitrary digits.
* **TF-IDF Feature Extraction:** Uses unigrams & bigrams (`ngram_range=(1,2)`) with sublinear TF scaling to capture meaningful spam indicators.
* **Ensemble Learning (Soft Voting):** Combines three distinct algorithms to optimize accuracy and stability:
  * **LinearSVC** (Calibrated for probability estimates)
  * **Logistic Regression**
  * **Multinomial Naive Bayes**
* **Interactive Inference:** Allows quick CLI testing for custom input messages.

---

## Performance & Evaluation

The model achieves an overall **Accuracy of ~99%** on the test dataset (33% split).

| Class | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| **Ham** (Legitimate) | 0.99 | 1.00 | 0.99 |
| **Spam** | 0.98 | 0.94 | 0.96 |

---

## Tech Stack

* **Language:** Python 3.x
* **Data Processing:** `pandas`, `numpy`, `re`
* **Machine Learning:** `scikit-learn` (`TfidfVectorizer`, `VotingClassifier`, `LinearSVC`, `LogisticRegression`, `MultinomialNB`)

---

## Project Structure

```text
sms-spam-detection/
├── requirements.txt    # Python dependencies
├── models.py       # Main script (Preprocessing, Training, Inference)
└── README.md           # Project documentation
```
## Credits & Acknowledgments

* **Dataset:** The dataset used in this project is the **SMS Spam Collection Dataset**, sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection) and hosted by [Aman Kharwal](https://github.com/amankharwal).
* **Libraries:** Built with open-source tools including [Scikit-Learn](https://scikit-learn.org/), [Pandas](https://pandas.pydata.org/), and [NumPy](https://numpy.org/).