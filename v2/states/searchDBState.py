from pydantic import BaseModel, Field

class GradeRelevance(BaseModel):
    binary_score: str = Field(
        description="Is the retrieved content relevant to the query? 'yes' or 'no'"
    )