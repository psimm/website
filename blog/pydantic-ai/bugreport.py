import nest_asyncio
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

load_dotenv(dotenv_path=".env")


nest_asyncio.apply()


def get_temperature(ctx: RunContext) -> str:
    return "The temperature is 20 degrees Celsius."

def get_precipitation(ctx: RunContext) -> str:
    return "The precipitation is 10mm."


agent_free = Agent(
    model="openai:gpt-4o-mini",
    tools=[get_temperature, get_precipitation],
)

result_free = agent_free.run_sync("What's the weather?")
print(result_free.data)


class StructuredResponse(BaseModel):
    temperature: str
    precipitation: str


agent_structured = Agent(
    model="openai:gpt-4o-mini",
    result_type=StructuredResponse,
    tools=[get_temperature, get_precipitation],
)

result_structured = agent_structured.run_sync("What's the weather?")
print(result_structured.data)
