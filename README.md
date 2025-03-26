# Titanic Survival Prediction

## 📌 Project Overview
This project aims to predict the survival of passengers aboard the Titanic using machine learning models. The dataset consists of passenger details such as age, sex, ticket class, and other relevant features. By applying classification algorithms, we determine the likelihood of survival based on these attributes.

## 📂 Dataset
The dataset used in this project is the **Titanic dataset**, which contains information about passengers, including:
- `PassengerId`: Unique ID of each passenger
- `Survived`: Target variable (1 = Survived, 0 = Did not survive)
- `Pclass`: Ticket class (1st, 2nd, 3rd)
- `Name`: Passenger's name
- `Sex`: Gender
- `Age`: Age of the passenger
- `SibSp`: Number of siblings/spouses aboard
- `Parch`: Number of parents/children aboard
- `Ticket`: Ticket number
- `Fare`: Ticket price
- `Cabin`: Cabin number
- `Embarked`: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## 🛠 Preprocessing Steps
To prepare the dataset for model training, the following preprocessing steps were applied:
1. **Handling Missing Values**:
   - Filled missing `Age` values with the median age.
   - Filled missing `Fare` values with the median fare.
   - Filled missing `Embarked` values with the most frequent category.
2. **Encoding Categorical Variables**:
   - Converted `Sex` and `Embarked` columns to numerical values using label encoding.
3. **Feature Selection**:
   - Selected relevant features: `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`.
4. **Data Splitting**:
   - Split the dataset into 80% training and 20% testing sets.
5. **Feature Scaling**:
   - Standardized numerical features using `StandardScaler` to improve model performance.

## 🤖 Model Selection
Several classification models were implemented to predict survival:
- **Logistic Regression**
- **Random Forest Classifier**
- **Gradient Boosting Classifier**
- **AdaBoost Classifier**
- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**
- **Decision Tree Classifier**

The models were trained and evaluated based on their accuracy, precision, recall, and F1-score.

## 📊 Performance Evaluation
Each model's performance was evaluated using:
- **Accuracy Score**
- **Classification Report** (Precision, Recall, F1-Score)
- **Confusion Matrix** (Visualization of predictions)

After evaluation, the best-performing model was selected based on accuracy. The model was then used for final predictions.

## 🚀 How to Run the Project
To execute this project, follow these steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/Deekshu966/Titanic-Survival-Prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Titanic-Survival-Prediction
   ```
3. Install required dependencies:
   
4. Open the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
5. Run the notebook `Titanic.ipynb` to preprocess the data, train models, and evaluate results.

## 📂 Output
- **Model Accuracy Scores** (Sorted from highest to lowest)
- **Best Performing Model**
- **Confusion Matrix for each model**
- **Test predictions saved as `test_predictions.csv`**

## 🔥 Conclusion
This project successfully applies machine learning techniques to predict Titanic survival outcomes. Through preprocessing, feature engineering, and model selection, we achieved a high-performing classifier for survival prediction.

---

**🔗 Repository:** [GitHub Link](https://github.com/Deekshu966/Titanic-Survival-Prediction)

