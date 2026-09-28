import os
import certifi


import requests

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults


from langchain.agents import (
    create_react_agent,
    AgentExecutor
)

from langchain import hub

# ==========================================
# LOAD ENV VARIABLES
# ==========================================
os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")

# ==========================================
# SEARCH TOOL
# ==========================================

search_tool = TavilySearchResults(max_results=2)

# ==========================================
# WEATHER TOOL
# ==========================================

@tool
def get_weather_data(city: str) -> str:
    """
    Fetch current weather information for a city.
    """
    if not WEATHERSTACK_API_KEY:
        return "Weatherstack API key is missing. Set WEATHERSTACK_API_KEY in the environment or .env file."

    try:
        response = requests.get(
            "https://api.weatherstack.com/current",
            params={"access_key": WEATHERSTACK_API_KEY, "query": city},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as error:
        return f"Weather service request failed ({type(error).__name__})."
    except ValueError:
        return "Weather service returned an invalid JSON response."

    if not isinstance(data, dict):
        return "Weather service returned an unexpected response format."

    if "error" in data:
        error = data["error"]
        return (
            f"Weatherstack API error ({error.get('code')}): "
            f"{error.get('info', 'Unknown API error')}"
        )

    if "current" not in data:
        return f"Weatherstack returned no current conditions for {city}."

    return (
        f"City: {city}\n"
        f"Temperature: {data['current']['temperature']}°C\n"
        f"Weather: {data['current']['weather_descriptions'][0]}\n"
        f"Humidity: {data['current']['humidity']}%"
    )

# ==========================================
# LLM
# ==========================================

llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0,
    groq_api_key=GROQ_API_KEY
)

# ==========================================
# PROMPT
# ==========================================

prompt = hub.pull("hwchase17/react")

# ==========================================
# TOOLS
# ==========================================

tools = [
    search_tool,
    get_weather_data
]

# ==========================================
# CREATE AGENT
# ==========================================

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# ==========================================
# EXECUTOR
# ==========================================

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

# ==========================================
# RUN
# ==========================================

response = agent_executor.invoke({
    "input": (
        "Find the capital of India"
        "and then find the current weather of hyderabad."
    )
})

print("\n========================")
print("FINAL OUTPUT")
print("========================\n")

print(response["output"])