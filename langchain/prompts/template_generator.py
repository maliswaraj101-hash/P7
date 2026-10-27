from langchain_core.prompts import PromptTemplate


template = PromptTemplate(
    template = """is this city {city_input} in this {country_input} ?
""",
input_variables = ['city_input' , 'country_input']
)

template.save('template.json')