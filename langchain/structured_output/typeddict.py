from typing import TypedDict , Annotated , Literal

class Person(TypedDict):
    name : str
    age : int

new_person : Person = {"name" : "ssss" , "age" : 22}
# this will just tell the user that name should be string , even if i pass an int in name , it wont give an error , typed dict does not do validation , it just tells what the class variable type should be.


# practical use

model = ChatOpenAI()

structured_model = model.with_structured_output(Person)

result = structured_model.invoke(""" my name is swaraj and i am 21 years old""")


print(result)

# the result will be a dict
#  result = { 'name' : 'swaraj' , 'age' : 21}
# this will be our output

# we can also do this

class Person2(TypedDict):
    name : Annotated[str , "name of the user"]
    age : Annotated[int , "age of the user"]
# u can also have options
    size : Annotated[Literal['medium' , 'large'] , "give the size"]


