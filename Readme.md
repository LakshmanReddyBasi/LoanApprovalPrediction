# Loan Approval Prediction Model

## Problem Statement

This project is about building a computer model that can predict if a loan application will be approved or rejected. This helps banks make quicker and better decisions, and also manage their risks. We used a dataset with various details about loan applicants and their requests to build this model.

## Approach Followed

We followed these steps to build our prediction model:

1.  **Getting the Data Ready:**
    * Loaded the `financial_lending.csv` dataset.
    * Removed any extra, unused columns.
    * Did a quick check to see what kind of data we had and if anything was missing.

2.  **Handling Missing Information:**
    * Found all the spots where data was missing.
    * For text-based missing data (like 'gender'), we filled it in with the most common answer.
    * For number-based missing data (like 'income'), we filled it in with the middle value (median).

3.  **Turning Text into Numbers:**
    * Converted text categories (like 'male/female', 'married/not married', 'education levels', 'self-employed status', 'residence area') into numerical codes using a `LabelEncoder`.
    * Changed the 'approval_status' from 'Y' (Yes) and 'N' (No) into 1s and 0s.

4.  **Exploring and Adjusting Data:**
    * Looked at graphs (histograms) of number data to see their spread and if they were lopsided.
    * Made lopsided income and loan amount data more balanced using a math trick (`np.log1p`).
    * Used bar charts (count plots) to see how many people fell into different categories.
    * Created a heatmap to see how all the features related to each other, and especially to loan approval. We noticed that having a credit history was a really big deal for getting a loan.

5.  **Scaling Numbers:**
    * Adjusted all numerical data (like income, loan amount, credit history) so they were on a similar scale. This helps our models learn better.

6.  **Splitting Data for Training and Testing:**
    * Divided our data into a training part (80% for the model to learn from) and a testing part (20% to check how well it learned).
    * Made sure the split kept the same proportion of approved/rejected loans in both parts, which is important because we had more approved loans than rejected ones.


7. **Training and Checking Models:**
    * We tried out several different machine learning models:
        * Logistic Regression (our basic model, and one we adjusted using SMOTE)
        * Decision Tree (a simpler tree-like model)
        * Random Forest (a more powerful model made of many trees)
        * Gradient Boosting (another strong tree-based model)
        * XGBoost (an even more advanced version of Gradient Boosting)
    * We checked how well each model performed using its overall accuracy, a detailed report (showing precision, recall, and F1-score for approvals and rejections), and a confusion matrix (a table showing correct and incorrect predictions).

9.  **Fine-Tuning Models:**
    * We used `GridSearchCV` to find the best settings for our Logistic Regression and XGBoost models. This helps them perform their best.

## Dataset Used

* **File Name:** `financial_lending.csv`
* **What it is:** This file holds all the information we used about people applying for loans. It includes things like their gender, if they're married, how many people depend on them, their education, if they're self-employed, where they live, their income, the loan amount they asked for, the loan term, and their credit history. And, of course, whether their loan was approved or not.

## Model Details

Here’s a quick look at the models we tried:

* **Logistic Regression:** This is a simple, straightforward model good for 'yes/no' predictions. We saw how it worked normally and also how it improved when we helped it with the uneven loan data using SMOTE.
* **Decision Tree:** This model makes decisions like a flowchart. It gave us a baseline but showed it struggled a bit with the uneven data.
* **Random Forest:** This is like having a "committee" of many decision trees. It performed much better than a single tree and we tuned it for even better results.
* **Gradient Boosting:** Another advanced model that builds trees one after another, trying to fix the mistakes of the previous one.
* **XGBoost:** A super optimized and popular version of Gradient Boosting. We spent time fine-tuning this one, and it turned out to be the best.

## Results Achieved

After all the testing, the **Tuned XGBoost Classifier** was our best model. Here’s a summary of how our top models performed:

| Metric                          | Tuned XGBoost | Tuned Random Forest | Logistic Regression (with SMOTE) |
| :------------------------------ | :------------ | :------------------ | :------------------------------- |
| **Overall Accuracy**            | **86.18%**    | 85.37%              | 85.37%                           |
| **For Rejected Loans (Class 0)**|               |                     |                                  |
| Precision (Correctly Rejected)  | **96%**       | 88%                 | 95%                              |
| Recall (Caught All Rejected)    | 58%           | **61%**             | 55%                              |
| F1-score (Balance)              | 72%           | **72%**             | 70%                              |
| **For Approved Loans (Class 1)**|               |                     |                                  |
| Precision (Correctly Approved)  | 84%           | **85%**             | 83%                              |
| Recall (Caught All Approved)    | **99%**       | 96%                 | **99%**                          |
| F1-score (Balance)              | 91%           | 90%                 | 90%                              |

**What We Found:**
* The **Tuned XGBoost model** was the winner with the highest overall accuracy at **86.18%**.
* It was really good at **correctly identifying rejected loans (96% precision)**, meaning it rarely said 'yes' to a bad loan.
* It also almost perfectly identified **approved loans (99% recall)**, so it didn't miss many good applicants.
* The **Tuned Random Forest** was also very strong, especially good at catching a slightly higher percentage of all actual rejected loans (61% recall).
* **Having a credit history** was by far the most important factor for deciding loan approval across all models, followed closely by income and the loan amount requested.

## Pending Improvements

Here are some ideas for what could be done next to make the model even better:

1.  **Better Handling of Uneven Data:** We could try applying the SMOTE trick (or similar methods) more consistently to all models, even during the fine-tuning process, to see if it helps them learn even better from the smaller group of rejected loans.
2.  **More Fine-Tuning:** We only tested a few settings for our models. We could try a wider range of settings to potentially squeeze out even more performance.
3.  **Smarter Data Checking:** Use more advanced ways to split and check the data (like stratified k-fold cross-validation) to get a more reliable idea of how well our models truly perform.
4.  **Other Ways to Measure Success:** Look at other graphs and numbers like ROC AUC or Precision-Recall curves. These are especially useful for uneven datasets to see how good the model is at telling the two groups apart.
5.  **Creating New Features:** We could invent new useful pieces of information from the existing data, like calculating a 'debt-to-income' ratio, which might help the model more.
6.  **Combining Models:** Try mixing different models together (like stacking or voting) to see if their combined power can beat any single model.
7.  **Trying Deep Learning:** If we had a much larger dataset, we could explore using neural networks, which are very powerful models.
8.  **Explaining Complex Models:** For models like XGBoost, it's sometimes hard to see *why* they made a certain decision. Tools like SHAP or LIME could help us understand their inner workings better.
