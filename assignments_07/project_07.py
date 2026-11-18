"""Part2 mini-project: World Happiness Agent (smolagents CodeAgent)."""
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from scipy import stats
from smolagents import CodeAgent, OpenAIServerModel, tool
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for the agent's worker thread


# Load OPENAI_API_KEY from the .env file (searches parent folders too)
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Paths are relative to this file so the script works from any folder
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "happiness_project"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Shared global DataFrame used by all tools
df = None

COLUMN_NAMES = {
    "Country": "country",
    "Regional indicator": "region",
    "Happiness score": "happiness_score",
    "Ladder score": "happiness_score",  # the 2024 file uses this name
    "GDP per capita": "gdp_per_capita",
}


@tool
def load_happiness_data() -> dict:
    """Load the World Happiness dataset (2015-2024) into memory.

    Reads every yearly CSV file, standardizes the column names, adds a
    'year' column, and merges them into one table. Call this first, before
    any other tool.

    Returns:
        A dict with "shape" (rows, columns) and "columns" (list of column names).
    """
    global df
    frames = []
    for path in sorted(DATA_DIR.glob("world_happiness_*.csv")):
        year_df = pd.read_csv(path, sep=";", decimal=",")
        year_df = year_df.rename(columns=COLUMN_NAMES)
        year_df["year"] = int(path.stem.split("_")[-1])
        frames.append(year_df)

    if not frames:
        return {"error": f"No world_happiness_*.csv files found in {DATA_DIR}"}

    df = pd.concat(frames, ignore_index=True)
    return {"shape": list(df.shape), "columns": list(df.columns)}


@tool
def summarize_column(column: str) -> dict:
    """Return descriptive statistics for a single column in the loaded dataset.

    Gives count, mean, std, min, quartiles, and max for numeric columns.

    Args:
        column: Exact name of the column to summarize, e.g. "happiness_score".

    Returns:
        A dict of descriptive statistics, or {"error": "..."} if no data is
        loaded or the column does not exist.
    """
    if df is None:
        return {"error": "No data loaded. Call load_happiness_data first."}
    if column not in df.columns:
        return {"error": f"Column '{column}' not found. Available: {list(df.columns)}"}
    return df[column].describe().to_dict()


@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation coefficient and p-value between two numeric columns.

    Rows with missing values in either column are ignored.

    Args:
        col1: Name of the first numeric column, e.g. "gdp_per_capita".
        col2: Name of the second numeric column, e.g. "happiness_score".

    Returns:
        A dict with "col1", "col2", "pearson_r", and "p_value" (rounded to 4
        decimals), or {"error": "..."} on bad input.
    """
    if df is None:
        return {"error": "No data loaded. Call load_happiness_data first."}
    for col in (col1, col2):
        if col not in df.columns:
            return {"error": f"Column '{col}' not found. Available: {list(df.columns)}"}
    try:
        pair = df[[col1, col2]].dropna()
        r, p = stats.pearsonr(pair[col1], pair[col2])
    except Exception as e:
        return {"error": f"Could not compute correlation: {e}"}
    return {"col1": col1, "col2": col2,
            "pearson_r": round(float(r), 4), "p_value": round(float(p), 4)}


@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """Return the top N countries ranked by a given column for a specific year.

    Args:
        column: Column to rank by, e.g. "happiness_score".
        year: The year to filter on, between 2015 and 2024.
        n: How many countries to return. Defaults to 5.

    Returns:
        A dict with key "top_countries" holding a list of dicts, each with
        "country" and the requested column value, or {"error": "..."} on bad input.
    """
    if df is None:
        return {"error": "No data loaded. Call load_happiness_data first."}
    if column not in df.columns:
        return {"error": f"Column '{column}' not found. Available: {list(df.columns)}"}
    year_df = df[df["year"] == year]
    if year_df.empty:
        return {"error": f"No data for year {year}."}
    top = year_df.sort_values(column, ascending=False).head(n)
    return {"top_countries": top[["country", column]].to_dict(orient="records")}


SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.
Use the available tools for loading data, summarizing columns, computing correlations,
and ranking countries. Write Python code directly only when the tools are not sufficient
(for example, when creating custom plots or computing something the tools don't cover).
Be concise and student-friendly in your responses.

IMPORTANT: the tools do not give you a DataFrame, so any code you write must first build
one. Do not use placeholders such as pd.DataFrame(), and do not call load_happiness_data
inside your code to get data. Copy this block exactly at the start of your code:

import pandas as pd
import matplotlib.pyplot as plt
frames = []
for y in range(2015, 2025):
    d = pd.read_csv(DATA_DIR + "/world_happiness_" + str(y) + ".csv", sep=";", decimal=",")
    d = d.rename(columns={"Country": "country", "Regional indicator": "region",
                          "Happiness score": "happiness_score", "Ladder score": "happiness_score",
                          "GDP per capita": "gdp_per_capita"})
    d["year"] = y
    frames.append(d)
data = pd.concat(frames, ignore_index=True)
data = data.dropna(subset=["region"])

Then do your analysis on `data`. When you are finished, always call final_answer(...)
with a short summary. Do not call matplotlib.use; the backend is already set.
""" + f'\nDATA_DIR = "{DATA_DIR}"\n'

model = OpenAIServerModel(api_key=api_key, model_id="gpt-4o-mini")

agent = CodeAgent(
    tools=[load_happiness_data, summarize_column, compute_correlation, get_top_n_countries],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "scipy.stats"],
    max_steps=8,
)


if __name__ == "__main__":
    # Task 3: guided queries
    queries = [
        "Load the happiness data and tell me its shape and column names.",
        "Summarize the happiness_score column.",
        "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
        "Show me the top 5 happiest countries in 2020.",
        f"Plot happiness_score over the years as a line chart, with one line per region. "
        f"Save the plot to {OUTPUT_DIR / 'happiness_by_region.png'}.",
    ]
    for query in queries:
        print(f"\n--- Query: {query} ---")
        response = agent.run(query, reset=False)
        print(response)

       my_query_1 = "Which 3 countries had the highest average happiness_score across all years?"
    response_1 = agent.run(my_query_1, reset=False)
    print(response_1)
    # Comment: Code generation only. No tool was called. get_top_n_countries ranks
    # countries for a single year, so it can't average across all years. The agent wrote
    # its own pandas code, data.groupby('country')['happiness_score'].mean().nlargest(3),
    # and returned Finland (7.67), Denmark (7.58), and Iceland (7.52).

    my_query_2 = ("Which region's average happiness_score increased the most "
                  "from 2015 to 2024?")
    response_2 = agent.run(my_query_2, reset=False)
    print(response_2)
    # Comment: Code generation only. No tool was called. None of my four tools compare
    # years or group by region. The agent filtered the data to 2015 and 2024, took the
    # mean score per region for each year, subtracted them, and used idxmax() to find
    # the largest increase: Central and Eastern Europe, about +0.66.


# --- Reflection ---
#
# 1. In Query 3, the agent called compute_correlation and reported a Pearson r of
#    0.6313 and a p-value of 0.0, along with the label "significant." That is
#    accurate, but the answer was thin. It returned only a small dictionary and
#    never explained what the numbers mean. It did not say that r = 0.63 is a
#    moderately strong positive relationship (richer countries tend to report
#    higher happiness), and it did not say how confident we can be in it. The
#    p-value of 0.0 is also just the tool's rounding to 4 decimals; the real
#    value is below 0.0001, not exactly zero. The agent also did not mention that
#    the data pools ten years of observations for the same countries, so the rows
#    are not fully independent, and that correlation does not show GDP causes
#    happiness.
#
# 2. Yes, I was surprised by how badly the agent handled passing data between its
#    tools and its own code. When Query 5 first asked for a plot, the tools only
#    return small summaries, and the agent could not see the DataFrame. It
#    repeatedly wrote pd.DataFrame(happiness_data) on the tool's dictionary,
#    created empty placeholder DataFrames, or hit KeyError: 'year' because it
#    never added the year column. It also guessed the year range as 2015-2021 and
#    tried to import glob, which is not an allowed import. On the other hand, it
#    was better than I expected at recovering from some errors: when DATA_DIR was
#    undefined, it defined it and continued. I also noticed that in Query 1 it
#    wrote its own pandas code instead of calling load_happiness_data, so it
#    reported a shape of (1499, 11) instead of the tool's (1502, 11).
#
# 3. A tool like get_grouped_average(column, group_by) would help. It would return
#    the mean of a column grouped by one or more fields, such as year and region.
#    That would answer questions like "Which region's average happiness score
#    increased the most from 2015 to 2024?" or "How has the average score in
#    Western Europe changed each year?" with a tool call, so the agent would not
#    have to rebuild the DataFrame in its own code, which is where it kept
#    failing.