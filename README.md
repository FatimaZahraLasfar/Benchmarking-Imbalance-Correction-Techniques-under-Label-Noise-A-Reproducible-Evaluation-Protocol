# Benchmarking Imbalance Correction Techniques under Label Noise: A Reproducible Evaluation Protocol

This repository contains the code and experiments for a systematic evaluation of common imbalance correction techniques, particularly focusing on their performance and robustness in the presence of label noise. The project emphasizes the importance of a standardized, reproducible evaluation protocol to draw reliable conclusions in machine learning research.

The experiments are conducted on the "Credit Card Fraud Detection" dataset, which is highly imbalanced and serves as a representative case for this study.

## Dataset

All experiments use the **Credit Card Fraud Detection** dataset.

*   **Source**: [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
*   **Characteristics**:
    *   Instances: 284,807 transactions
    *   Features: 30 (V1-V28 are PCA components, plus Time and Amount)
    *   Imbalance: Highly imbalanced, with only 492 frauds (0.172%). The natural ratio is approximately 1:578.

For the experiments, subsets of this data are created to simulate specific imbalance ratios (e.g., 1:50, 1:100) in a controlled manner.

## Hypotheses Tested

This project is structured around four central hypotheses, each tested in a dedicated Jupyter Notebook.

### H1: SMOTE vs. SMOTE+Tomek
*   **Notebook**: `test_H1_creditcard_improved.ipynb`
*   **Hypothesis**: *Hybrid resampling methods (SMOTE + Tomek Links) achieve a significantly higher F1-score (≥ +5%) on the minority class compared to SMOTE alone in highly imbalanced contexts (≥ 1:50).*
*   **Protocol**: A 1:50 imbalance ratio is simulated. A RandomForestClassifier is evaluated using 5-fold stratified cross-validation. The primary metric is the F1-score on the fraud class, with statistical significance determined by a Wilcoxon signed-rank test.
*   **Conclusion**: **Hypothesis Rejected**. For this dataset, the PCA-transformed features result in well-separated classes. The Tomek Links phase found no ambiguous pairs to remove, making SMOTETomek's performance identical to SMOTE's. The experiment shows that the theoretical benefit of hybrid methods may not be observable on pre-processed, well-separated data.

### H2: Class Weighting vs. SMOTE
*   **Notebook**: `test_H2_creditcard_improved.ipynb`
*   **Hypothesis**: *Cost-sensitive learning (class weighting) offers more stable performance in terms of PR-AUC (standard deviation ≤ 0.03) than oversampling methods (SMOTE) in extremely imbalanced situations (≥ 1:100).*
*   **Protocol**: A 1:100 imbalance ratio is simulated. A LogisticRegression model is evaluated using PR-AUC over 5-fold stratified cross-validation. Stability is measured by the standard deviation of scores across folds.
*   **Conclusion**: **Hypothesis Rejected**. Both methods exhibited statistically indistinguishable variance in their PR-AUC scores (confirmed by a Levene test, p=0.94). While the hypothesis was not confirmed, the experiment demonstrates that on this dataset, both methods are equally stable.

### H3: Sensitivity to Label Noise
*   **Notebook**: `test_H3_creditcard_improved.ipynb`
*   **Hypothesis**: *Oversampling methods like SMOTE are more sensitive to label noise (PR-AUC degradation > 15%), while adaptive methods like ADASYN or hybrid methods like SMOTE+Tomek show better robustness (degradation ≤ 8%).*
*   **Protocol**: Progressive levels of label noise (0% to 20%) are injected into the training data's majority class (asymmetric noise). The degradation in PR-AUC is measured for SMOTE, ADASYN, and SMOTE+Tomek.
*   **Conclusion**: **Hypothesis Rejected**. All tested oversampling methods proved to be highly vulnerable to label noise. The hypothesized robustness of ADASYN and SMOTE+Tomek was not observed. The study highlights that label noise significantly undermines the performance of these techniques, and none of them are inherently robust to it under the tested conditions.

### H4: Standardized vs. Non-Standardized Evaluation Protocol
*   **Notebook**: `test_H4_creditcard_improved.ipynb`
*   **Hypothesis**: *Using a standardized experimental protocol (stratified CV, fixed seeds, appropriate metrics like PR-AUC) reveals statistically significant differences that non-standardized protocols (simple KFold, no seeds, accuracy metric) would otherwise miss.*
*   **Protocol**: Two protocols are compared. The **standardized** protocol uses `StratifiedKFold`, fixed `random_state` for all components, and evaluates using F1-score and PR-AUC. The **non-standardized** protocol uses simple `KFold`, no fixed seeds, and evaluates using only `accuracy`.
*   **Conclusion**: **Hypothesis Partially Rejected, but its Premise is Vindicated**. The standardized protocol did not find a significant difference (p > 0.05), but it provided a clear, reliable, and reproducible answer: "SMOTE and ADASYN perform similarly on this task." In contrast, the non-standardized protocol produced misleadingly high accuracy scores (~99.8%) that masked any real differences and gave results that varied with each run. This experiment powerfully demonstrates that a rigorous protocol is essential for valid scientific conclusions, even when the conclusion is one of no significant difference.

## Repository Structure

*   `test_H1_creditcard_improved.ipynb`: Experiment and analysis for Hypothesis 1.
*   `test_H2_creditcard_improved.ipynb`: Experiment and analysis for Hypothesis 2.
*   `test_H3_creditcard_improved.ipynb`: Experiment and analysis for Hypothesis 3.
*   `test_H4_creditcard_improved.ipynb`: Experiment and analysis for Hypothesis 4.

Each notebook is self-contained and includes data loading, preprocessing, model evaluation, statistical testing, and visualization generation for its corresponding hypothesis.

## How to Run the Experiments

### Prerequisites

You will need Python 3.8+ and the following libraries:
*   `pandas`
*   `numpy`
*   `scikit-learn`
*   `imbalanced-learn`
*   `matplotlib`
*   `seaborn`
*   `scipy`
*   `jupyter`

You can install them using pip:
```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn scipy jupyter
```

### Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/FatimaZahraLasfar/Benchmarking-Imbalance-Correction-Techniques-under-Label-Noise-A-Reproducible-Evaluation-Protocol.git
    cd Benchmarking-Imbalance-Correction-Techniques-under-Label-Noise-A-Reproducible-Evaluation-Protocol
    ```

2.  **Download the dataset:**
    Download the `creditcard.csv` file from the [Kaggle dataset page](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

3.  **Update the dataset path:**
    In each `.ipynb` notebook, locate the following line and update the path to where you saved `creditcard.csv`.
    ```python
    df = pd.read_csv("D:\\ENSET\\S2\\Méthodologie de Recherche\\Datasets\\creditcard.csv")
    ```
    For example, if you placed it in a `data` folder inside the repository directory:
    ```python
    df = pd.read_csv("data/creditcard.csv")
    ```

4.  **Run the notebooks:**
    Launch Jupyter Notebook or JupyterLab and open the notebooks to execute the experiments. All visualizations and results will be generated directly within the notebooks.
    ```bash
    jupyter notebook
