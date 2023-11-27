# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
# from sklearn.model_selection import train_test_split
import os
import pandas as pd
import streamlit as st


features = [' income_annum', ' cibil_score', ' loan_term', ' loan_amount']

data_file_path = os.getenv("LOAN_DATA_PATH")
df = pd.read_csv(data_file_path)

x = df[features]

y = df[' loan_status']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)

model.fit(x_train, y_train)


y_pred = model.predict(x_test)

# cm = confusion_matrix(y_test, y_pred)
# print(cm)


# print("Accuracy:", accuracy_score(y_test, y_pred))

st.header('Loan Approval Data Project')

st.image('images/CibilStatusScatter.PNG')

st.header('Enter the data')

income = st.text_input('Enter the income')

cibil = st.text_input('Enter the credit score')

term = st.text_input('Enter the loan duration')

amount = st.text_input('Enter the loan amount')

data = pd.DataFrame({
    ' income_annum': [income],
    ' cibil_score': [cibil],
    ' loan_term': [term],
    ' loan_amount':[amount]
})

st.table(data)

# info = pd.DataFrame({
#     ' income_annum': [5700000],
#     ' cibil_score': [382],
#     ' loan_term': [20],
#     ' loan_amount':[15000000]
# })



display = st.checkbox('Show Results', value=False)

if display:
    response = model.predict(data)


    if ' Approved' in response:
        st.write('Approved')
        
    else:
        st.write('Rejected')
    


# streamlit run StreamLit.py