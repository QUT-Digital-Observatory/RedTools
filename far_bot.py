#feasibilty bot for Ausreddit

from ollama import create

from api import AusRedditData
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware 
from langgraph.checkpoint.memory import InMemorySaver 
from langgraph.types import Command
from langchain.tools import tool 

model = ChatGoogleGenerativeAI(model="gemini-flash-latest")

@tool
def get_submission_aggregates(query, start, end, period):
    '''Count how many Reddit submissions match a search query over time.

Queries the RedTools API and returns the number of matching submissions
grouped into regular time bins (e.g. weekly or monthly counts). Use this
to understand trends in how often a topic is posted about on Reddit.

Uses GET /aggregates/submissions/.

Parameters:
-----------
query : str
    Keyword or phrase to search for in submission titles and text.
    Only submissions matching this query are counted.
start : int or str, optional
    Start of the date range to aggregate over. Accepts a Unix timestamp
    (int/float), 'yyyy-mm-dd', or 'dd/mm/yyyy'. Interpreted as
    00:00:00 UTC. Defaults to the earliest record in the database.
end : int or str, optional
    End of the date range to aggregate over. Accepts a Unix timestamp
    (int/float), 'yyyy-mm-dd', or 'dd/mm/yyyy'. Interpreted as
    23:59:59 UTC. Defaults to the latest record in the database.
period : str, optional
    Size of each time bin. One of 'day', 'week', 'month', or 'year'.
    Default is 'week'.

Returns:
--------
pd.DataFrame
    One row per time bin with columns:
    - start (datetime, UTC): bin start timestamp
    - end (datetime, UTC): bin end timestamp
    - frequency (int): number of matching submissions in that bin
'''
    
    endpoint = AusRedditData()
    results =endpoint.get_submission_aggregates(query, start, end, period)
    return results

@tool
def get_ngrams(query, start, end):
    '''Track how frequently one or more words or phrases appear in Reddit comments over time.

Queries the RedTools API and returns monthly usage percentages for each
ngram (1–3 word phrase). Use this to compare the relative popularity of
terms or topics across time.

Uses POST /aggregates/ngrams/.

Each query must be 1–3 words. The first (and only the first) query may
contain a '*' wildcard as a whole word, in which case the API expands
it to the top 5 matching ngrams and returns them as separate columns.

Parameters:
-----------
queries : list[str]
    One or more ngram strings (1–3 words each) to look up. The first
    entry may use '*' as a wildcard word (e.g. 'climate *') to retrieve
    the top 5 matching completions as separate series.
start : str or int, optional
    Start of the date range. Accepts 'yyyy-mm-dd', 'dd/mm/yyyy',
    'yyyy-mm', or an integer year. Year and month are extracted and
    sent to the API. Defaults to the earliest available data.
end : str or int, optional
    End of the date range. Same formats as start. Defaults to the
    most recent available data.

Returns:
--------
pd.DataFrame
    Index: monthly period labels (e.g. '2021-01'). One column per
    query (or per expanded wildcard ngram). Values are the percentage
    of total comments that month containing the ngram, rounded to
    4 decimal places.
'''
    endpoint = AusRedditData()
    results = endpoint.get_ngrams(query, start, end)
    return results


agent = create_agent(
    model = model,
    tools = [get_submission_aggregates, get_ngrams])
      