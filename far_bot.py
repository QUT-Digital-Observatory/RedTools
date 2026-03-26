#feasibilty bot for Ausreddit

import json
import os
import matplotlib.pyplot as plt
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

@tool
def plot_submission_frequency(data_json, query, output_path='submission_frequency.png'):
    '''Plot submission counts over time as a bar chart and save to a PNG file.

Takes the JSON output from get_submission_aggregates and produces a bar
chart showing how many matching submissions occurred in each time bin.
Call this after get_submission_aggregates to visualise frequency trends.

Parameters:
-----------
data_json : str
    JSON string returned by get_submission_aggregates.
query : str
    The search query, used as the chart title.
output_path : str, optional
    File path to save the PNG. Default is 'submission_frequency.png'.

Returns:
--------
str
    The path to the saved PNG file.
'''
    data = json.loads(data_json)
    if not data:
        return 'No data to plot.'
    labels = [row['start'][:10] for row in data]
    values = [row['frequency'] for row in data]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(labels, values)
    ax.set_title(f'Submission frequency: "{query}"')
    ax.set_xlabel('Period start')
    ax.set_ylabel('Submissions')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    return output_path

@tool
def plot_ngram_volume(data_json, output_path='ngram_volume.png'):
    '''Plot ngram usage percentages over time as a line chart and save to a PNG file.

Takes the JSON output from get_ngrams and produces a line chart showing
the percentage of comments mentioning each ngram per month. Call this
after get_ngrams to visualise relative volume trends.

Parameters:
-----------
data_json : str
    JSON string returned by get_ngrams.
output_path : str, optional
    File path to save the PNG. Default is 'ngram_volume.png'.

Returns:
--------
str
    The path to the saved PNG file.
'''
    data = json.loads(data_json)
    if not data:
        return 'No data to plot.'
    periods = [row['period'] for row in data]
    columns = [k for k in data[0].keys() if k != 'period']
    fig, ax = plt.subplots(figsize=(12, 4))
    for col in columns:
        ax.plot(periods, [row[col] for row in data], label=col, marker='o', markersize=3)
    ax.set_title('Ngram volume (% of comments)')
    ax.set_xlabel('Month')
    ax.set_ylabel('% of comments')
    ax.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    return output_path

@tool
def save_report(content, output_path='report.md'):
    '''Save the feasibility report to a markdown file.

Parameters:
-----------
content : str
    The markdown-formatted report text to save.
output_path : str, optional
    File path to save the report. Default is 'report.md'.

Returns:
--------
str
    The path to the saved file.
'''
    with open(output_path, 'w') as f:
        f.write(content)
    return output_path


agent = create_agent(
    model=model,
    tools=[get_submission_aggregates, get_ngrams, plot_submission_frequency, plot_ngram_volume, save_report],
    prompt="""You are a research assistant that assesses the feasibility of studying a topic \
using the AusReddit collection — a database of Australian Reddit submissions and comments.

When given a topic and time period, you will:
1. Call get_submission_aggregates and get_ngrams to retrieve the data.
2. Call plot_submission_frequency and plot_ngram_volume to generate charts.
3. Produce a feasibility report in markdown covering three dimensions:

## Occurrence
Is the topic present in the collection during the time period? When does it first and last appear?

## Frequency
How often does it appear? Describe trends over time (peaks, growth, decline) using the submission counts.

## Volume
How much of the overall conversation does it represent? Describe its relative presence using the ngram percentages.

If the user asks to save the report, call save_report with the markdown text.
Be concise and factual. Ground all claims in the data returned by the tools."""
)


def run(query, start=None, end=None, save=False):
    """Run a feasibility assessment for a topic and optional date range.

    Parameters:
    -----------
    query : str
        The topic to assess (e.g. 'bluey').
    start : str, optional
        Start of the date range. Accepts 'yyyy-mm-dd' or 'dd/mm/yyyy'.
    end : str, optional
        End of the date range. Accepts 'yyyy-mm-dd' or 'dd/mm/yyyy'.
    save : bool, optional
        If True, the agent will save the report and charts to files.
    """
    parts = [f'Assess the feasibility of studying "{query}" in the AusReddit collection']
    if start or end:
        parts.append(f'for the period {start or "the beginning"} to {end or "the present"}')
    if save:
        parts.append('Save the report and charts to files named after the topic.')
    prompt = ' '.join(parts) + '.'

    result = agent.invoke({'messages': [{'role': 'user', 'content': prompt}]})
    print(result['messages'][-1].content)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Feasibility assessment bot for the AusReddit collection.')
    parser.add_argument('query', help='Topic to assess (e.g. "bluey")')
    parser.add_argument('--start', help='Start date (yyyy-mm-dd or dd/mm/yyyy)', default=None)
    parser.add_argument('--end', help='End date (yyyy-mm-dd or dd/mm/yyyy)', default=None)
    parser.add_argument('--save', action='store_true', help='Save the report and charts to files')
    args = parser.parse_args()
    run(args.query, args.start, args.end, args.save)
