from typing import Annotated,Sequence, TypedDict
import os
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from langchain_core.tools import tool
from geopy.geocoders import Nominatim
from pydantic import BaseModel, Field
import requests

from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig

from langgraph.graph import StateGraph, END
from langchain_tavily import TavilySearch

from datetime import datetime

load_dotenv()

os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

class AgentState(TypedDict):
    """The state of the agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    number_of_steps: int

geolocator = Nominatim(user_agent="weather-app")

class SearchInput(BaseModel):
    location:str = Field(description="The city and state, e.g., San Francisco")
    date:str = Field(description="the forecasting date for when to get the weather format (yyyy-mm-dd)")

class TavilySchema(BaseModel):
    query : str = Field(description="Query on the basis of which search is being done")

@tool("get_weather_forecast", args_schema=SearchInput, return_direct=True)
def get_weather_forecast(location: str, date: str):
    """Retrieves the weather using Open-Meteo API for a given location (city) and a date (yyyy-mm-dd). Returns a list dictionary with the time and temperature for each hour."""
    location = geolocator.geocode(location)
    if location:
        try:
            response = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={location.latitude}&longitude={location.longitude}&hourly=temperature_2m&start_date={date}&end_date={date}")
            data = response.json()
            return {time: temp for time, temp in zip(data["hourly"]["time"], data["hourly"]["temperature_2m"])}
        except Exception as e:
            return {"error": str(e)}
    else:
        return {"error": "Location not found"}
    
@tool("tavily_search", args_schema=TavilySchema, return_direct=True)
def tavily_search(query : str):
    """This function helps to make a search using Tavily and get the information about a particular query!!"""
    tool = TavilySearch(max_results=2)
    res = tool.invoke(query)
    return {"response" : res}

@tool
def add(a : int, b : int):
    """A function which adds 2 numbers"""
    return a + b

@tool
def multiply(a : int, b : int):
    """A function which multiplies 2 numbers"""
    return a*b

@tool
def subtract(a : int, b : int):
    """A function which subtracts tqo numbers"""
    return a - b

@tool 
def division(a : float, b : float):
    """A function which divides two numbers"""
    return round(a/b, 2)

tools = [get_weather_forecast, tavily_search, add, multiply, subtract, division]

# Create LLM class
llm = ChatGoogleGenerativeAI(model= "gemini-1.5-pro")

# Bind tools to the model
model = llm.bind_tools(tools)
tools_by_name = {tool.name: tool for tool in tools}

# Define our tool node
def call_tool(state: AgentState):
    outputs = []
    for tool_call in state["messages"][-1].tool_calls:
        # Get the tool by name
        tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
        outputs.append(
            ToolMessage(
                content=tool_result,
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    return {"messages": outputs}

def call_model(state: AgentState, config: RunnableConfig,):
    response = model.invoke(state["messages"], config)
    return {"messages": [response]}


def should_continue(state: AgentState):
    messages = state["messages"]
    if not messages[-1].tool_calls:
        return "end"
    return "continue"

# Define a new graph with our state
graph_builder = StateGraph(AgentState)

# 1. Add our nodes 
graph_builder.add_node("llm", call_model)
graph_builder.add_node("tools",  call_tool)
# 2. Set the entrypoint as `agent`, this is the first node called
graph_builder.set_entry_point("llm")
# 3. Add a conditional edge after the `llm` node is called.
graph_builder.add_conditional_edges(
    "llm",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)

graph_builder.add_edge("tools", "llm")
graph = graph_builder.compile()

while True:
    user_input = input("Write your query? ")
    if user_input == "exit":
        print("Good Bye")
        break
    else:
        inputs = {"messages" : [("user", user_input)]}
        for state in graph.stream(inputs, stream_mode="values"):
            last_message = state["messages"][-1]
            last_message.pretty_print()


