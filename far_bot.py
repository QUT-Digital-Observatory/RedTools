#feasibilty bot for Ausreddit

import json
from api import AusRedditData
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.tools import tool

endpoint = AusRedditData()
model = ChatGoogleGenerativeAI(model="gemini-flash-latest")

@tool
def get_submission_aggregates(query, start=None, end=None, period='week'):
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
str (JSON)
    JSON array, one object per time bin with keys:
    - start (ISO 8601 UTC string): bin start timestamp
    - end (ISO 8601 UTC string): bin end timestamp
    - frequency (int): number of matching submissions in that bin
'''
    df = endpoint.get_submission_aggregates(query, start, end, period)
    if df.empty:
        return json.dumps([])
    df['start'] = df['start'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    df['end'] = df['end'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    return df.to_json(orient='records')

@tool
def get_ngrams(queries, start=None, end=None):
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
str (JSON)
    JSON array, one object per month with a 'period' key (e.g. '2021-01')
    and one key per query (or expanded wildcard ngram) whose value is the
    percentage of total comments that month containing the ngram, rounded
    to 4 decimal places.
'''
    df = endpoint.get_ngrams(queries, start, end)
    if df.empty:
        return json.dumps([])
    df = df.reset_index()
    return df.to_json(orient='records')


agent = create_agent(
    model=model,
    tools=[get_submission_aggregates, get_ngrams])
