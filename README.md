🏠 California Housing Price Prediction

A complete Machine Learning regression project that predicts housing prices in California using real-world census data.
This project demonstrates the full end-to-end data science pipeline from data preprocessing to model evaluation.

📌 Project Overview

The goal of this project is to build a model that can accurately predict the median house value in different districts of California based on various features such as:

Income levels
Population
Housing age
Number of rooms & bedrooms
Geographic location (latitude & longitude)

This project is inspired by real-world real estate price prediction systems used in the industry.

🧠 Key Concepts Covered
Data cleaning & preprocessing
Handling missing values
Exploratory Data Analysis (EDA)
Feature engineering
Train/test splitting
Regression modeling
Model evaluation
Hyperparameter tuning (if included)
📊 Dataset

The dataset is based on the 1990 California census and includes:

longitude
latitude
housing_median_age
total_rooms
total_bedrooms
population
households
median_income
median_house_value (target)
⚙️ Machine Learning Workflow
Load dataset
Handle missing values
Perform exploratory data analysis
Feature scaling / transformation
Train machine learning models
Evaluate performance
Improve model accuracy
🤖 Models Used

Common models used in this type of project:

Linear Regression
Decision Tree Regressor
Random Forest Regressor
Gradient Boosting (optional improvement)
🛠️ Technologies Used
Python 🐍
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
Jupyter Notebook
📂 Project Structure
California-Housing-Price-Prediction/
│
├── clofornia houses.ipynb   # Main notebook
├── data/                    # Dataset files (if included)
├── images/                  # Visualizations
└── README.md
🚀 How to Run
1. Clone repository
git clone https://github.com/amirafarhan/California-Housing-Price-Prediction.git
2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn
3. Run Jupyter Notebook
jupyter notebook
4. Open file
clofornia houses.ipynb
📈 Results

The model is able to:

Learn patterns between location, income, and housing prices
Predict median house values with reasonable accuracy
Show feature importance (if Random Forest is used)
🔮 Future Improvements
Deploy model using Flask / Streamlit
Add interactive map visualization (Geo plots)
Use advanced models (XGBoost / LightGBM)
Feature scaling optimization
Hyperparameter tuning with GridSearchCV
👨‍💻 Author

Amira Farhan
Machine Learning & Computer Vision Engineer

⭐ Support

If you like this project:

⭐ Star the repository
🍴 Fork it
🚀 Share it
