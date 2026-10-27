from dataclasses import Field
from typing import Optional
from pydantic import BaseModel


class Student(BaseModel):
    name : str
    # name : str = 'swaraj' ; this can be set as the default value
    # name : Optional[str] = None
    # if name is not provided , it will be set as None making the name as optional.
    # cgpa : float = Field(gt = 0 , lt = 10 , default=5 , description = 'give the cgpa of the student')
    # cgpa should be greater than 0 and less than 10 , this way we can add constraints.


new_student = {'name' : 'swaraj'}

student = Student(**new_student)

print(student)

# if do 'name' : 32 , it will give a validation error , as name is supposed to be string.