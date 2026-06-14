import pandas as pd

# Load Dataset
df = pd.read_excel("WEEK 2/Dataset 2.xlsx")

# Q1
print("Q1 - First 5 Records")
print(df.head())

# Q2
print("\nQ2 - Rows and Columns")
print(df.shape)

# Q3
print("\nQ3 - Column Names")
print(df.columns)

# Q4
print("\nQ4 - Numerical Features")
print(df.select_dtypes(include=['int64', 'float64']).columns)

print("\nCategorical Features")
print(df.select_dtypes(include=['object']).columns)

# Q5
print("\nQ5 - Missing Values")
print(df.isnull().sum())

# Q6
print("\nQ6 - Average Age")
print(df['Age'].mean())

# Q7
print("\nQ7 - Average Watch Hours Per Week")
print(df['WatchHoursPerWeek'].mean())

# Q8
print("\nQ8 - Average Monthly Spending")
print(df['MonthlySpend'].mean())

# Q9
print("\nQ9 - Users in Each Subscription Category")
print(df['SubscriptionType'].value_counts())

# Q10
renewal_percentage = (
    (df['SubscriptionRenewed'] == 'Yes').sum()
    / len(df)
) * 100

print("\nQ10 - Subscription Renewal Percentage")
print(f"{renewal_percentage:.2f}%")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Q11 - Convert Categorical Features into Numerical Form

df_encoded = df.copy()

label_encoder = LabelEncoder()

categorical_columns = [
    'Gender',
    'SubscriptionType',
    'FavoriteGenre',
    'SubscriptionRenewed'
]

for col in categorical_columns:
    df_encoded[col] = label_encoder.fit_transform(df_encoded[col])

print("\nQ11 - Encoded Dataset")
print(df_encoded.head())


# Q12 - Define Features (X) and Target (y)

X = df_encoded.drop('SubscriptionRenewed', axis=1)
y = df_encoded['SubscriptionRenewed']

print("\nQ12 - Features Shape")
print(X.shape)

print("\nTarget Shape")
print(y.shape)


# Q13 - Split Dataset into Training and Testing Sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nQ13 - Training Set Shape")
print(X_train.shape)

print("\nTesting Set Shape")
print(X_test.shape)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Q14 - Train Decision Tree Model

dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)

print("\nQ14 - Decision Tree Model Trained Successfully")


# Q15 - Accuracy

dt_accuracy = accuracy_score(y_test, y_pred_dt)

print("\nQ15 - Decision Tree Accuracy")
print(dt_accuracy)


# Q16 - Confusion Matrix

cm = confusion_matrix(y_test, y_pred_dt)

print("\nQ16 - Confusion Matrix")
print(cm)

from sklearn.neighbors import KNeighborsClassifier

# Q17 - Train KNN Classifier

knn_model = KNeighborsClassifier(n_neighbors=5)

knn_model.fit(X_train, y_train)

y_pred_knn = knn_model.predict(X_test)

print("\nQ17 - KNN Model Trained Successfully")


# Q18 - Compare Accuracy

knn_accuracy = accuracy_score(y_test, y_pred_knn)

print("\nQ18 - KNN Accuracy")
print(knn_accuracy)

print("\nDecision Tree Accuracy")
print(dt_accuracy)

if knn_accuracy > dt_accuracy:
    print("\nKNN performed better.")
elif dt_accuracy > knn_accuracy:
    print("\nDecision Tree performed better.")
else:
    print("\nBoth models performed equally.")

    from sklearn.linear_model import LinearRegression

# Q19 - Train Linear Regression Model

X_reg = df_encoded.drop('MonthlySpend', axis=1)
y_reg = df_encoded['MonthlySpend']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg,
    y_reg,
    test_size=0.2,
    random_state=42
)

lr_model = LinearRegression()

lr_model.fit(X_train_reg, y_train_reg)

print("\nQ19 - Linear Regression Model Trained Successfully")


# Q20 - Predict Monthly Spending for a New User

new_user = [[
    751,  # UserID
    25,   # Age
    1,    # Gender
    1,    # SubscriptionType
    20,   # WatchHoursPerWeek
    2,    # DevicesUsed
    2,    # FavoriteGenre
    5,    # AdClicks
    1     # SubscriptionRenewed
]]

predicted_spend = lr_model.predict(new_user)

print("\nQ20 - Predicted Monthly Spending")
print(predicted_spend[0])

