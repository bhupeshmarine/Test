import json
import pandas as pd

from itertools import product
from typing import List

from pydantic import BaseModel, Field
from databricks_langchain import ChatDatabricks
from langchain_core.prompts import ChatPromptTemplate

class PairScore(BaseModel):
    pair_id: str
    score: float = Field(ge=0, le=1)


class ComparisonResult(BaseModel):
    comparisons: List[PairScore]


llm = ChatDatabricks(
    endpoint="databricks-claude-sonnet-4-5",
    temperature=0
)

structured_llm = llm.with_structured_output(ComparisonResult)


prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are an entity matching judge.

        Compare every provided pair and return a similarity score
        between 0 and 1.

        1 = definitely the same entity
        0 = definitely different

        Consider spelling differences, abbreviations, initials,
        formatting differences, word order, and minor typos.

        Return one score for every pair_id.
        """
    ),
    (
        "human",
        "{pairs}"
    )
])


judge_chain = prompt | structured_llm



def compare_fields(df, list_1, list_2):

    result_df = df.copy()

    # Create all combinations
    column_pairs = list(product(list_1, list_2))

    # Create output columns
    output_columns = {
        f"{left}__{right}":
        f"score__{left}__vs__{right}"

        for left, right in column_pairs
    }

    # Store results
    all_results = []

    for _, row in result_df.iterrows():

        pairs = []

        for left, right in column_pairs:

            pair_id = f"{left}__{right}"

            pairs.append({
                "pair_id": pair_id,
                "left_value": str(row[left]),
                "right_value": str(row[right])
            })

        # One LLM call for the entire row
        response = judge_chain.invoke({
            "pairs": json.dumps(pairs)
        })

        row_scores = {
            item.pair_id: item.score
            for item in response.comparisons
        }

        all_results.append(row_scores)

    # Add score columns
    for pair_id, output_column in output_columns.items():

        result_df[output_column] = [
            row_result.get(pair_id)
            for row_result in all_results
        ]

    return result_df
