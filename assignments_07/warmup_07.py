# ============================================================
# Imports and setup
# ============================================================
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from scipy.stats import pearsonr
import os
from smolagents import tool, ToolCallingAgent, CodeAgent, OpenAIServerModel
import matplotlib
matplotlib.use("Agg")              # non-interactive backend: safe in worker threads
import matplotlib.pyplot as plt

load_dotenv()          # reads OPENAI_API_KEY from the .env file
client = OpenAI()

RESOURCES_DIR = Path("resources")   # relative: run this script from inside assignments_07/
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

# --- Lesson 02: Tool Definitions and the ReAct Loop ---

# Q1
def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert a Celsius temperature to Fahrenheit and return it as a formatted string."""
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}°C is {fahrenheit}°F"

# JSON schema: the model never sees the Python code, only this description,
# so the description fields are how it decides when and how to use the tool.
celsius_to_fahrenheit_schema = {
    "type": "function",
    "function": {
        "name": "celsius_to_fahrenheit",
        "description": "Convert a Celsius temperature to Fahrenheit and return it as a formatted string.",
        "parameters": {
            "type": "object",
            "properties": {
                "celsius": {
                    "type": "number",
                    "description": "The temperature in degrees Celsius to convert.",
                }
            },
            "required": ["celsius"],
        },
    },
}

# Direct calls (no agent yet). -40 is where Celsius and Fahrenheit are equal.
for temp in [0, 100, -40]:
    print(celsius_to_fahrenheit(temp))


# Q2
def get_current_time() -> str:
    '''Return the current local time as a formatted string.'''
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# The lesson's tools list: one tool only. run_agent reads this global.
tools = [
    {
        'type': 'function',
        'function': {
            'name': 'get_current_time',
            'description': 'Returns the current local time as a string.',
            'parameters': {'type': 'object', 'properties': {}, 'required': []},
        },
    }
]

def run_agent(user_prompt: str) -> str:
    '''Run a minimal ReAct-style agent for a single user prompt.'''

    SYSTEM_PROMPT = '''You are a simple assistant that can tell the current time.
                     Use the tool get_current_time whenever a user asks about the time.'''

    # Step 1: start the conversation with system and user messages
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]

    # Step 2: first API call - the model decides whether to call a tool
    first_response = client.chat.completions.create(
        model='gpt-4.1-mini',
        messages=messages,
        tools=tools,
        tool_choice='auto',  # model chooses whether to use a tool
    )
    print("First response received from model...")
    print(first_response)
    first_message = first_response.choices[0].message

    # Record what the model said so far
    messages.append(
        {
            'role': 'assistant',
            'content': first_message.content,
            'tool_calls': first_message.tool_calls,
        }
    )

    # Step 3: check if the model requested any tools
    if first_message.tool_calls:
        print("Agentic mode engaged...")
        for tool_call in first_message.tool_calls:
            function_name = tool_call.function.name
            # In this example we only have one tool: get_current_time
            if function_name == 'get_current_time':
                tool_result = get_current_time()
            else:
                tool_result = f'Error: unknown tool {function_name}.'

            print('Tool called:', function_name)
            print('Tool result:', tool_result)

            # Step 3b: append the tool output so the model can see it
            messages.append(
                {
                    'role': 'tool',
                    'tool_call_id': tool_call.id,
                    'name': function_name,
                    'content': tool_result,
                }
            )

        # Step 4: second API call - model sees the tool result and gives final answer
        second_response = client.chat.completions.create(
            model='gpt-4.1-mini',
            messages=messages,
        )
        print("Second response received from model...")
        print(second_response)
        final_message = second_response.choices[0].message
        return final_message.content or ''
    else:
        print("No tools needed....")

    # If there were no tool calls, the first response was already the final answer
    return first_message.content or ''

# --- PREDICTION (written BEFORE running) ---
# 1. Tool call? No. The only tool is get_current_time, which has nothing to do
#    with temperature conversion, and celsius_to_fahrenheit is not registered,
#    so the model can't see or use it. I expect it to answer from its own knowledge.
# 2. API calls? One. With no tool call, the first response is the final answer
#    and the second API call is skipped.

result = run_agent("Convert 100 degrees Celsius to Fahrenheit")
print(result)

# --- RESULT ---
# Prediction was correct. The model did not call a tool ("No tools needed...."
# printed). It explained the formula F = (9/5) * C + 32 itself and answered
# 212 degrees Fahrenheit. celsius_to_fahrenheit is defined in this file, but
# the agent can't use it unless it is passed in the tools list.


# Q3
# Change 1: add the Q1 schema to the tools list so the model can see both tools.
tools = tools + [celsius_to_fahrenheit_schema]

def run_agent(user_prompt: str) -> str:
    '''Same as Q2, but dispatches both tools.'''

    SYSTEM_PROMPT = '''You are a simple assistant that can tell the current time
                     and convert Celsius temperatures to Fahrenheit.
                     Use the available tools when they apply.'''

    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]

    first_response = client.chat.completions.create(
        model='gpt-4.1-mini',
        messages=messages,
        tools=tools,
        tool_choice='auto',
    )
    first_message = first_response.choices[0].message

    messages.append(
        {
            'role': 'assistant',
            'content': first_message.content,
            'tool_calls': first_message.tool_calls,
        }
    )

    if first_message.tool_calls:
        print("Agentic mode engaged...")
        for tool_call in first_message.tool_calls:
            function_name = tool_call.function.name

            # Change 2: dispatch the new tool. Arguments arrive as a JSON
            # string, so parse them before calling the Python function.
            if function_name == 'get_current_time':
                tool_result = get_current_time()
            elif function_name == 'celsius_to_fahrenheit':
                args = json.loads(tool_call.function.arguments)
                tool_result = celsius_to_fahrenheit(args['celsius'])
            else:
                tool_result = f'Error: unknown tool {function_name}.'

            print('Tool called:', function_name)
            print('Tool result:', tool_result)

            messages.append(
                {
                    'role': 'tool',
                    'tool_call_id': tool_call.id,
                    'name': function_name,
                    'content': tool_result,
                }
            )

        second_response = client.chat.completions.create(
            model='gpt-4.1-mini',
            messages=messages,
        )
        return second_response.choices[0].message.content or ''
    else:
        print("No tools needed....")

    return first_message.content or ''


response_a = run_agent("What is 37 degrees Celsius in Fahrenheit?")
print("Response A:", response_a)
# Tool called: YES (celsius_to_fahrenheit). The question is a temperature
# conversion, which matches the tool's description, so the model requested
# the tool, Python ran it, and the model used the result in its final answer.
# (Confirm against the "Tool called:" debug line in the output.)

response_b = run_agent("What is the boiling point of water in plain English?")
print("Response B:", response_b)
# Tool called: NO ("No tools needed...." printed). This is a general knowledge
# question; neither tool is relevant, so the model answered from its own
# knowledge.


# --- Lesson 03: Multi-Tool Agent ---

# Q4
class CsvManager:
    def __init__(self, resources_dir: Path):
        self.resources_dir = resources_dir
        self.df = None
        self.csv_name = None

    # --- Small internal helpers --------------------------------------

    def _normalize_csv_name(self, filename: str) -> str:
        if not filename.lower().endswith(".csv"):
            return filename + ".csv"
        return filename

    def _available_csv_files(self) -> list[str]:
        if not self.resources_dir.exists():
            return []
        return sorted(
            [
                p.name
                for p in self.resources_dir.iterdir()
                if p.is_file() and p.suffix.lower() == ".csv"
            ]
        )

    def _ensure_loaded(self):
        if self.df is None:
            files = self._available_csv_files()
            example = files[0] if files else "your_file.csv"
            return {
                "error": (
                    "No CSV is loaded yet. First load one from resources/. "
                    f"For example: load_csv '{example}'."
                )
            }
        return None

    # --- Tools (public methods) --------------------------------------

    def list_csv_files(self):
        """
        List available CSV files in resources/.
        """
        files = self._available_csv_files()
        if not files:
            return {
                "message": (
                    "No CSV files found in resources/. "
                    "Create a resources/ folder and put one or more .csv files inside it."
                ),
                "files": [],
            }
        return {"files": files}

    def load_csv(self, filename: str):
        """
        Load a CSV file from resources/ and make it the active dataset.

        filename can be "bike_commute" or "bike_commute.csv".
        """
        filename = self._normalize_csv_name(filename)
        path = self.resources_dir / filename

        if not path.exists():
            return {
                "error": f"Could not find '{filename}' in resources/.",
                "available_files": self._available_csv_files(),
            }

        self.df = pd.read_csv(path)
        self.csv_name = filename

        return {
            "message": f"Loaded {filename} with shape {self.df.shape}.",
            "columns": self.df.columns.tolist(),
        }

    def get_columns(self):
        """
        Return column names for the currently loaded CSV.
        """
        error = self._ensure_loaded()
        if error:
            return error
        return self.df.columns.tolist()

    def summarize_columns(self, columns: list[str] | None = None):
        """
        Return basic summary stats for one or more columns.

        If columns is None, summarize all columns.
        Uses pandas.describe(include="all") to stay simple and readable.
        """
        error = self._ensure_loaded()
        if error:
            return error

        if columns is None:
            data = self.df
        else:
            missing = [c for c in columns if c not in self.df.columns]
            if missing:
                return {"error": f"These columns are not in the data: {missing}"}
            data = self.df[columns]

        summary = data.describe(include="all").transpose().round(3)
        return summary.to_dict()

    def describe_column(self, column: str):
        """
        Simple summary for a single column using pandas.describe().
        """
        error = self._ensure_loaded()
        if error:
            return error

        if column not in self.df.columns:
            return {"error": f"'{column}' is not a column. Options: {self.df.columns.tolist()}"}

        s = self.df[column]
        summary = s.describe().to_dict()

        cleaned = {}
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                cleaned[key] = round(value, 3)
            else:
                cleaned[key] = value

        return cleaned

    # >>> Q4: NEW METHOD, added inside the class <<<
    def compute_correlation(self, col1: str, col2: str):
        """
        Compute the Pearson correlation between two columns in the loaded DataFrame.
        Returns the correlation coefficient and p-value.
        """
        # Reuse the lesson's helper: returns an error dict if no CSV is loaded.
        error = self._ensure_loaded()
        if error:
            return error

        # Both columns must exist. Listing the real names helps the agent retry.
        for col in (col1, col2):
            if col not in self.df.columns:
                return {"error": f"Column '{col}' not found. Available columns: {self.df.columns.tolist()}"}

        # pearsonr can't handle NaN, so drop rows missing either value.
        data = self.df[[col1, col2]].dropna()

        try:
            r, p = pearsonr(data[col1], data[col2])
        except Exception as e:
            # e.g. non-numeric column or too few rows
            return {"error": f"Could not compute correlation: {e}"}

        # float() makes numpy values JSON-serializable.
        return {
            "col1": col1,
            "col2": col2,
            "pearson_r": round(float(r), 4),
            "p_value": round(float(p), 4),
        }

    def plot_data(self, y: str, x: str | None = None, plot_type: str = "line"):
        """
        Plot from the active CSV.

        - If x is None: plot y vs row index.
        - If x is provided: plot y vs x.
        """
        error = self._ensure_loaded()
        if error:
            return error

        if plot_type not in ["scatter", "line"]:
            return "Error: I can only do 'scatter' or 'line'."

        if y not in self.df.columns:
            return f"Error: column '{y}' is not in {self.df.columns.tolist()}"

        # If someone accidentally passes x == y, treat it like "plot y"
        if x == y:
            x = None

        # Scatter needs x
        if plot_type == "scatter" and x is None:
            return "Error: scatter plots need both x and y columns."

        title_csv = self.csv_name or "current CSV"

        # CHANGED from the lesson: plt.show() crashes on macOS when smolagents
        # runs the tool in a worker thread, so I save the figure to outputs/
        # instead and return the file path.
        if x is None:
            ax = self.df[y].plot(kind="line")
            ax.set_title(f"{title_csv} | Line plot: {y} vs row index")
            out_path = OUTPUTS_DIR / f"{y}_vs_index_line.png"
            plt.savefig(out_path)
            plt.close()
            return f"Plotted {y} vs row index as a line plot. Saved to {out_path}."

        if x not in self.df.columns:
            return f"Error: column '{x}' is not in {self.df.columns.tolist()}"

        ax = self.df.plot(x=x, y=y, kind=plot_type)
        ax.set_title(f"{title_csv} | {plot_type.title()} plot: {y} vs {x}")
        out_path = OUTPUTS_DIR / f"{y}_vs_{x}_{plot_type}.png"
        plt.savefig(out_path)
        plt.close()

        return f"Plotted {y} vs {x} as a {plot_type}. Saved to {out_path}."
   

csv_manager = CsvManager(resources_dir=RESOURCES_DIR)

# Q4: schema entry for the new tool
compute_correlation_schema = {
    "type": "function",
    "function": {
        "name": "compute_correlation",
        "description": "Compute the Pearson correlation coefficient and p-value between two numeric columns of the currently loaded CSV.",
        "parameters": {
            "type": "object",
            "properties": {
                "col1": {"type": "string", "description": "Name of the first column."},
                "col2": {"type": "string", "description": "Name of the second column."},
            },
            "required": ["col1", "col2"],
        },
    },
}

tools_schema = [
    {"type": "function", "function": {
        "name": "list_csv_files",
        "description": "List available CSV files in resources/.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "load_csv",
        "description": "Load a CSV file from resources/ and make it the active dataset.",
        "parameters": {"type": "object", "properties": {
            "filename": {"type": "string", "description": "e.g. bike_commute or bike_commute.csv"}},
            "required": ["filename"]}}},
    {"type": "function", "function": {
        "name": "get_columns",
        "description": "Return column names for the currently loaded CSV.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "summarize_columns",
        "description": "Return summary stats for selected columns (or all columns).",
        "parameters": {"type": "object", "properties": {
            "columns": {"type": "array", "items": {"type": "string"},
                        "description": "Columns to summarize. Omit for all."}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "describe_column",
        "description": "Describe a single column (basic stats).",
        "parameters": {"type": "object", "properties": {
            "column": {"type": "string", "description": "Column name."}},
            "required": ["column"]}}},
    {"type": "function", "function": {
        "name": "plot_data",
        "description": "Plot from the active CSV as a line or scatter plot.",
        "parameters": {"type": "object", "properties": {
            "y": {"type": "string", "description": "Column for the y-axis."},
            "x": {"type": "string", "description": "Column for the x-axis (optional)."},
            "plot_type": {"type": "string", "enum": ["line", "scatter"]}},
            "required": ["y"]}}},
    compute_correlation_schema,   # Q4: new tool registered
]

node_tools = {
    "list_csv_files": csv_manager.list_csv_files,
    "load_csv": csv_manager.load_csv,
    "get_columns": csv_manager.get_columns,
    "summarize_columns": csv_manager.summarize_columns,
    "describe_column": csv_manager.describe_column,
    "plot_data": csv_manager.plot_data,
    "compute_correlation": csv_manager.compute_correlation,   # Q4: new entry
}

MAX_ROUNDS = 6   # tool-round limit

def run_agent_cycle(messages: list, user_input: str) -> str:
    # Appends to the list passed in, so Q6 can print the full history.
    messages.append({"role": "user", "content": user_input})
    for _ in range(MAX_ROUNDS):
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            tools=tools_schema,
        )
        msg = response.choices[0].message
        messages.append(msg)
        if not msg.tool_calls:                  # no tool requested -> final answer
            return msg.content
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments)
            result = node_tools[call.function.name](**args)
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result),  # tool content must be a string
            })
    return "Stopped: tool-round limit reached."


SYSTEM_PROMPT = (
    "You are a small data assistant for CSV files stored in resources/. "
    "Use the available tools to do any data work (do not guess). "
    "If no CSV is loaded yet, load one first (or list available CSV files). "
    "Keep answers short and student-friendly."
)


# Q5
# Prediction (written BEFORE running): in the lesson this request hit the
# tool-round limit because no correlation tool existed. Now compute_correlation
# is in tools_schema and node_tools, so I expect: load_csv -> compute_correlation
# -> final answer, without hitting the limit. I expect a negative pearson_r
# (more traffic density -> lower speed).

messages = [{"role": "system", "content": SYSTEM_PROMPT}]
result = run_agent_cycle(
    messages,
    "Load bike_commute.csv and compute the correlation between avg_traffic_density and avg_speed_kmh.",
)
print(result)

# --- RESULT ---
# Prediction was correct. The agent succeeded and did not hit the tool-round
# limit. It called load_csv on bike_commute.csv (160 rows, 6 columns), then
# compute_correlation on avg_traffic_density and avg_speed_kmh, then gave a
# final answer.
# Results: pearson_r = -0.5321, p_value = 0.0 (rounded to 4 decimals, so the
# true p-value is very small but not exactly zero).
# The sign matched my prediction: the correlation is negative, so higher
# traffic density goes with lower average speed. The strength is moderate
# (|r| about 0.53), so traffic density explains a good part of the variation
# in speed but far from all of it. With 160 rides, the tiny p-value means this
# is very unlikely to be chance. This is a correlation, not proof that traffic
# causes the slowdown.
# It worked because compute_correlation is registered in both tools_schema and
# node_tools, so the agent had a tool for the step that caused the round-limit
# failure in the lesson.


# Q6
# Printing the full messages list after Q5 shows the whole ReAct loop as a
# conversation history. Each dictionary has a "role" that says who produced it:
#
# - system:    the instructions I gave the agent up front (SYSTEM_PROMPT). It sets
#              the agent's behavior (use tools, don't guess, keep answers short).
#              It is always first and is not part of the reasoning loop itself.
# - user:      my request ("Load bike_commute.csv and compute the correlation...").
#              This is the task the agent is trying to solve.
# - assistant: the model's turns. In the loop, these are either a tool call
#              request (the "Act" step: which tool to run and with what
#              arguments) or the final natural-language answer. Tool-call
#              messages have no text content, just the tool_calls.
# - tool:      the result of running a tool, sent back to the model (the
#              "Observation" step). It is linked to the assistant's request by
#              tool_call_id. The model reads it to decide whether to call
#              another tool or give a final answer.
#
# Actual order from my run: system -> user -> assistant (load_csv call) ->
# tool (load result) -> assistant (compute_correlation call) -> tool
# (r = -0.5321, p = 0.0) -> assistant (final answer). This matched my
# expectation. The assistant messages appear as long strings instead of
# dictionaries because they are message objects from the OpenAI library, and
# json.dumps(default=str) converts them to text. Tool-call messages have
# content=None; the real content is in tool_calls. The final assistant
# message has tool_calls=None, which is what ends the loop.

print(json.dumps(messages, indent=2, default=str))
# --- Lesson 04: smolagents ---

# Q7
# Re-wrapping compute_correlation as a smolagents tool. The body just calls
# the CsvManager method from Q4, so the actual logic is not duplicated.
# smolagents builds the tool's description from the function itself, so the
# type hints and the docstring (including the Args section) matter here.
@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation between two numeric columns of the loaded CSV.

    Args:
        col1: Name of the first column.
        col2: Name of the second column.

    Returns:
        A dict with col1, col2, pearson_r and p_value (rounded to 4 decimals),
        or an error dict if no CSV is loaded or a column is not found.
    """
    return csv_manager.compute_correlation(col1, col2)

print(compute_correlation.description)
print(compute_correlation.inputs)   # extra: the argument info smolagents extracted

# COMPARISON: smolagents-generated tool vs. my hand-written Q4 JSON schema
#
# What print(compute_correlation.description) showed:
#   "Compute the Pearson correlation between two numeric columns of the loaded CSV."
# That is only the first line of my docstring. The argument information was
# stored separately in compute_correlation.inputs:
#   {'col1': {'type': 'string', 'description': 'Name of the first column.'},
#    'col2': {'type': 'string', 'description': 'Name of the second column.'}}
#
# Compared with Q4, where I wrote the JSON schema by hand (name, description,
# parameters with types and descriptions, and a "required" list), smolagents
# produced the same information automatically:
#   - tool name: from the function name
#   - description: from the docstring summary line
#   - argument types ('string'): from the type hints (col1: str, col2: str)
#   - argument descriptions: from the "Args:" section of the docstring
# The argument descriptions match the ones I wrote by hand in Q4, because I
# reused the same wording in the docstring.
# One difference is that I did not write a "required" list; smolagents works
# that out from the function signature (no default values means required).
#
# What smolagents needs from me, the developer, to produce a good description:
#   1. Type hints on every argument (col1: str), because they become the
#      argument types.
#   2. A clear docstring summary line saying what the tool does, because it
#      becomes the description the model reads when choosing a tool.
#   3. An "Args:" section with a description for every parameter, because
#      those become the argument descriptions the model uses to fill in values.
#   4. A "Returns:" section, which tells the model (and readers) what comes
#      back from the tool.
# The quality of the generated description depends on the quality of the
# docstring and type hints. If I write a vague or incomplete docstring, the
# agent gets a vague tool definition and may pick the wrong tool or pass bad
# arguments. In Q4 that same effort went into the JSON by hand; here it goes
# into the function's docstring and signature instead.

# Q8
# Name note: Q5 already uses SYSTEM_PROMPT for the Lesson 03 agent, so the
# smolagents instructions get their own name to avoid overwriting it.

# --- The lesson's @tool wrappers (each one calls the shared csv_manager) ---
@tool
def list_csv_files() -> dict:
    """List available CSV files in resources/.

    Returns:
        A dict with a "files" list, or a message if none are found.
    """
    return csv_manager.list_csv_files()


@tool
def load_csv(filename: str) -> dict:
    """Load a CSV file from resources/ and make it the active dataset.

    Args:
        filename: CSV filename in resources/. You can pass "bike_commute" or "bike_commute.csv".

    Returns:
        A dict with a status message and column names, or an error dict.
    """
    return csv_manager.load_csv(filename)


@tool
def get_columns() -> list[str] | dict:
    """Return column names for the currently loaded CSV.

    Returns:
        A list of column names, or an error dict if no CSV is loaded.
    """
    return csv_manager.get_columns()


@tool
def summarize_columns(columns: list[str] | None = None) -> dict:
    """Return summary stats for selected columns (or all columns).
    This includes count, mean, std, min, max, and percentiles for numeric columns,
    or count, unique, top, freq for categorical columns.

    Args:
        columns: Column names to summarize. If None, summarizes all columns.

    Returns:
        A dict of summary statistics (from pandas.describe), or an error dict.
    """
    return csv_manager.summarize_columns(columns)


@tool
def describe_column(column: str) -> dict:
    """Describe a single column (basic stats) for the requested column.
    This includes count, mean, std, min, max, and percentiles for numeric column,
    or count, unique, top, freq for categorical column.

    Args:
        column: The name of the column to describe.

    Returns:
        A dict of basic stats for the column, or an error dict.
    """
    return csv_manager.describe_column(column)


@tool
def plot_data(y: str, x: str | None = None, plot_type: str = "line") -> str | dict:
    """Plot from the active CSV.

    Args:
        y: Column name to plot on the y-axis.
        x: Column name to plot on the x-axis. If None, use row index.
        plot_type: "line" or "scatter". Scatter requires x and y.

    Returns:
        Generates and shows the plot.
        Returns a short success message string, or an error dict/string.
    """
    return csv_manager.plot_data(y=y, x=x, plot_type=plot_type)


# The same TOOLS list for both agents, including the Q7 compute_correlation tool.
TOOLS = [
    list_csv_files,
    load_csv,
    get_columns,
    summarize_columns,
    describe_column,
    plot_data,
    compute_correlation,   # added from Q7
]

# One model shared by both agents.
# (Use the lesson's api_key setup if it differs; this reads the same .env key.)
model = OpenAIServerModel(
    model_id="gpt-4o-mini",
    api_key=os.environ["OPENAI_API_KEY"],
)

SMOL_INSTRUCTIONS = (
    "You are a small data assistant to help analyze files stored in resources/. "
    "Use the available tools to do any work requested (do not guess). "
    "Keep answers short and student-friendly."
)

# NOTE: the lesson defines its own CODE_INSTRUCTIONS for the CodeAgent. Replace
# this with the lesson's text if you have it. This is a stand-in.
CODE_INSTRUCTIONS = (
    SMOL_INSTRUCTIONS
    + " You may write Python code. A CsvManager named csv_manager is available "
      "as an argument, with the loaded DataFrame in csv_manager.df. "
      "When you make a plot, save it with plt.savefig('outputs/<name>.png') "
      "and call plt.close(); do not use plt.show()."
)

tool_agent = ToolCallingAgent(
    tools=TOOLS,
    model=model,
    instructions=SMOL_INSTRUCTIONS,
)

code_agent = CodeAgent(
    tools=TOOLS,
    model=model,
    instructions=CODE_INSTRUCTIONS,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "numpy"],
    max_steps=10,
)

prompt = "Load bike_commute.csv. Plot avg_heart_rate vs duration_min as a scatter plot with green dots."

response_tool = tool_agent.run(prompt)
response_code = code_agent.run(prompt, additional_args={"csv_manager": csv_manager})

print("ToolCallingAgent response:", response_tool)
print("CodeAgent response:", response_code)

# --- PREDICTION (written before running) ---
# The plot_data tool only accepts y, x and plot_type. It has no color
# parameter, so I expect the ToolCallingAgent to make a scatter plot but not
# be able to make the dots green. I expect the CodeAgent to be able to,
# because it can write its own matplotlib code instead of being limited to
# the arguments of my tools.

# --- RESULTS ---
# 1. What each agent produced:
#
#    ToolCallingAgent: It called load_csv, then plot_data with
#      {'y': 'avg_heart_rate', 'x': 'duration_min', 'plot_type': 'scatter'}
#      and saved outputs/avg_heart_rate_vs_duration_min_scatter.png. It did NOT
#      change the dot color, because plot_data has no color parameter, so the
#      agent had no way to ask for green. The dots should be the pandas
#      default color (blue); I confirmed this by opening the PNG. [check this]
#      It also never told me it couldn't do the green part. Its final answer
#      said the scatter plot was created and saved, and quietly dropped the
#      color request. It also called plot_data a second time for no reason
#      (step 2) before giving the final answer.
#
#    CodeAgent: It wrote its own matplotlib code instead of calling plot_data:
#      plt.scatter(df['duration_min'], df['avg_heart_rate'], color='green'),
#      with a title, axis labels and a grid, then plt.savefig(...) to
#      outputs/avg_heart_rate_vs_duration.png and plt.close(). It finished in
#      two steps. The dots should be green; I confirmed this by opening the
#      PNG. [check this]
#      One thing to note: the CodeAgent did not load the CSV itself. It read
#      csv_manager.df, which was already filled in by the ToolCallingAgent's
#      earlier load_csv call, because both agents share the same csv_manager
#      object. If the CodeAgent had run first, df would have been None.
#
#    My prediction was correct: the ToolCallingAgent can only do what its tools
#    allow, and the CodeAgent can write new code to meet the request.
#
# 2. What this reveals about when each is more useful:
#    A ToolCallingAgent is best for fixed, well-defined jobs that my tools
#    already cover (load a file, summarize a column, compute a correlation).
#    It is predictable and can only run the tools I wrote, which is safer.
#    The downside is that anything my tools don't support, like dot color,
#    is silently left out, so I would need to add a color parameter to
#    plot_data, or notice the missing step myself.
#    A CodeAgent is more useful for flexible or open-ended requests, such as
#    styling a plot, combining steps, or doing something I didn't build a
#    tool for, because it can write the code on the spot. The trade-off is
#    that it is less predictable and executes generated code, so it needs
#    limits like authorized imports and max_steps, and I should check its
#    output instead of trusting the summary.
#    The two also differ in how they show their work: the ToolCallingAgent
#    shows a list of tool calls, and the CodeAgent shows the actual code it ran.

# Q9
#
# 1. A task where a ToolCallingAgent is the better choice:
#    Answering questions about a fixed dataset with a known set of operations,
#    for example: "Load bike_commute.csv and tell me the correlation between
#    avg_traffic_density and avg_speed_kmh." Or a support-style assistant that
#    looks up records and reports numbers to non-technical users.
#    What makes it a good fit: the task is narrow, well-defined and repeatable,
#    and every step maps onto a tool I already wrote and tested
#    (load_csv, compute_correlation, summarize_columns). The tools handle
#    validation (for example, compute_correlation returns an error dict if a
#    column is missing), so the results are consistent from run to run, and
#    the agent can only do what I allowed. It is also easy to audit, since the
#    log is a short list of tool calls with their arguments. A CodeAgent adds
#    flexibility the task doesn't need, along with extra unpredictability.
#
# 2. One meaningful risk of a CodeAgent that does not apply to a
#    ToolCallingAgent:
#    A CodeAgent writes Python code and then executes it, so the code is
#    generated by the model at run time and I have not reviewed it. That
#    means the model can do anything its code is allowed to do, not just
#    what my tools allow. For example, in Q8 the CodeAgent wrote its own
#    plt.savefig('outputs/...') call, which was harmless, but with broader
#    imports or a different prompt, generated code could read or overwrite
#    files, make network requests, or delete data. A wrong or malicious
#    prompt (or text hidden in a data file the agent reads) could steer it
#    into doing this. A ToolCallingAgent can only call the functions I
#    defined, with arguments checked against their schemas, so its worst case
#    is limited to those tools. This is why the CodeAgent needs guardrails
#    like additional_authorized_imports and max_steps, and why generated code
#    should run in a sandbox when the input is not trusted.
#    A smaller, related risk is that the output is harder to predict: in Q8
#    the CodeAgent also silently depended on csv_manager already being loaded
#    by the earlier agent, which would have failed if it had run first.