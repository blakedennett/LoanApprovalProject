
import streamlit as st

st.title("Loan Approval Website")
st.subheader("This is the subheader")
st.header('This is the header')


st.markdown("[Source Code](https://github.com/blakedennett/LoanApprovalProject)")

code_block = """
def func():
    print('hi')
"""
st.code(code_block, language='python')


# st.subheader('checking')

state = st.checkbox('Show Image', value=True, )

if state:
    st.image('images/AssetLoanTerm.PNG', width=700)

# st.markdown("""
# <style>
# .element-container.st-emotion-cache-e8g64f.e1f1d6gn3 
# {
#     color: red;
# }
# </style>""", unsafe_allow_html=True)

choice = st.radio("Select the place", options=['Las Vegas', 'Orlando'])
st.write(choice)

def disp():
    st.write('Hello there')

but = st.button("Display Greeting", on_click=disp)

select = st.selectbox("Select dog breed", options=('None Selected', 'Rotweiler', 'Border Collie'))

put = st.text_input('Enter the income')

st.write(put)


# streamlit run StPractice.py
