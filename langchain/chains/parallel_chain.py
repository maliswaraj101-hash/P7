from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel


# NOW WE WILL MAKE A APPLICATION WITH THIS FLOW
#                    Document
        #           /        \
        #      Module1       Module2
        #       /              \
        #    Notes             Quiz
        #       \              /
        #        \            /
        #         \          /
        #           Combined Model
        #                 |
        #               Merge
        #                 |
        #               Output


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "google/gemma-2-2b-it",
    huggingfacehub_api_token="YOUR_HF_TOKEN",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)


prompt1 = PromptTemplate(
    template = 'Generate short and simple notes from the following text \n {text} ',
    input_variables = ['text'],
)

prompt2 = PromptTemplate(
    template = 'Generate 5 short questions and answers from the following text \n {text} ',
    input_variables = ['text'],
)



prompt3 = PromptTemplate(
    template = 'Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz} ',
    input_variables = ['notes','quiz'],
)

parser = StrOutputParser()


parallel_chain = RunnableParallel({
    'notes' : prompt1 | model | parser,
    'quiz' : prompt2 | model | parser
})

merged_chain = prompt3 | model | parser

chain = parallel_chain | merged_chain

text = """
Linear Regression is a machine learning algorithm based on supervised learning. It performs a regression task. Regression models a target prediction value based on independent variables. It is mostly used for finding out the relationship between variables and forecasting. Different regression models differ based on – the kind of relationship between the dependent and independent variables, they are considering and the number of independent variables being used. This article is going to demonstrate how to use the various Python libraries to implement linear regression on a given dataset. We will demonstrate a binary linear model as this will be easier to visualize. In this demonstration, the model will use Gradient Descent to learn. You can learn about it here. 

Step 1: Importing all the required libraries 


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
Step 2: Reading the dataset:


df = pd.read_csv('https://www.kaggle.com/datasets/dftow001/bottle-csv')
df_binary = df[['Salnty', 'T_degC']]

# Taking only the selected two attributes from the dataset
df_binary.columns = ['Sal', 'Temp']
#display the first 5 rows
df_binary.head()
Output:

  

Step 3: Exploring the data scatter 


#plotting the Scatter plot to check relationship between Sal and Temp
sns.lmplot(x ="Sal", y ="Temp", data = df_binary, order = 2, ci = None)
plt.show()
Output:   



Step 4: Data cleaning 




# Eliminating NaN or missing input numbers
df_binary.fillna(method ='ffill', inplace = True)
  Step 5: Training our model 


X = np.array(df_binary['Sal']).reshape(-1, 1)
y = np.array(df_binary['Temp']).reshape(-1, 1)

# Separating the data into independent and dependent variables
# Converting each dataframe into a numpy array 
# since each dataframe contains only one column
df_binary.dropna(inplace = True)

# Dropping any rows with Nan values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

# Splitting the data into training and testing data
regr = LinearRegression()

regr.fit(X_train, y_train)
print(regr.score(X_test, y_test))
Output:

 

Step 6: Exploring our results 


y_pred = regr.predict(X_test)
plt.scatter(X_test, y_test, color ='b')
plt.plot(X_test, y_pred, color ='k')

plt.show()
# Data scatter of predicted values
"""
result = chain.invoke({'text' : text})

print(result)
