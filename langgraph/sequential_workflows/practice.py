from langgraph.graph import StateGraph , START , END
# from langchain_core.tools import tool
# from langchain_groq import ChatGroq
from typing import TypedDict , Literal
from langchain.output_parsers.structured import ResponseSchema


# @tool
# def name(name : str):
#     return name


# tools = [name]

# llm = ChatGroq(
#     api_key = "YOUR_GROQ_API_KEY",
#     model = "deepseek-r1-distill-llama-70b"
# )

# llm_with_tools = llm.bind_tools(tools)

# llm_with_tools.invoke()

class PracticeState(TypedDict):
    name : str
    age : int
    roll_number : int
    weight : float
    bmi : float


def register_student(state : PracticeState) -> PracticeState:
    state['name'] = "John Doe"
    state['age'] = 20
    state['roll_number'] = 12345
    return state


def calculate_bmi(state : PracticeState) -> PracticeState:
    weight = state['weight']
    bmi = weight / ((state['age'] / 100) ** 2)  # Simplified BMI calculation for demonstration

    return { "bmi" : bmi}

def check_bmi(state : PracticeState) -> Literal["underweight", "normal_weight", "overweight", "obese"]:
    bmi = state['bmi']

    if bmi < 18.5:
        return "underweight"
    elif 18.5 <= bmi < 24.9:
        return "normal_weight"
    elif 25 <= bmi < 29.9:
        return "overweight"
    else:
        return "obese"

def underweight(state : PracticeState) -> PracticeState:
    print("The student is underweight.")
    return state

def normal_weight(state : PracticeState) -> PracticeState:
    print("The student has a normal weight.")
    return state

def overweight(state : PracticeState) -> PracticeState: 
    print("The student is overweight.")
    return state

def obese(state : PracticeState) -> PracticeState:
    print("The student is obese.")
    return state

graph = StateGraph(PracticeState)

graph.add_node('register_student', register_student)
graph.add_node('calculate_bmi', calculate_bmi)
graph.add_node('check_bmi', check_bmi)
graph.add_node('underweight', underweight)
graph.add_node('normal_weight', normal_weight)
graph.add_node('overweight', overweight)
graph.add_node('obese', obese)

graph.add_edge(START, 'register_student')
graph.add_edge('register_student', 'calculate_bmi')
graph.add_conditional_edges('calculate_bmi', check_bmi)


graph.add_edge('underweight', END)
graph.add_edge('normal_weight', END)
graph.add_edge('overweight', END)
graph.add_edge('obese', END)

initial_state = {
    "name": "",
    "age": 0,
    "roll_number": 0,
    "weight": 70.0,  # Example weight in kg           
}

workflow = graph.compile()

result = workflow.invoke(initial_state)

print("Final State:", result)