
json_schema = {
    "title": "student",
    "description": "schema about students",
    "type": "object",
    "properties":{
        "name":"string",
        "age" : "integer"
    },
    "required":["name"]
}



model = ChatOpenAI()
    
structued_model = model.with_structured_output(json_schema)

result = structued_model.invoke(""" write something here which will be given as output by the model in the provided json schema , whatever the prompt this will be  , model will give it in structured output written above """)


# In invoke("..."), you write a normal prompt to the model (e.g., “Give me details of a student named John, 20 years old”).

# The model will then map its answer into your schema (in this case: JSON with "name" and "age").