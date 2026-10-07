
# =========================================================
# CUSTOMER RETENTION AI AGENT
# Streamlit Public Demo
# =========================================================

import sqlite3
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from sentence_transformers import SentenceTransformer, util

from google import genai
from google.genai import types


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Retention AI Agent",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("🤖 Customer Retention AI Agent")

st.write(
    "AI-powered telecom customer analysis using "
    "SQL, Machine Learning, Rule-Based Risk Analysis, "
    "RAG, and a Gemini AI Agent."
)

st.markdown("---")


# =========================================================
# DATASET
# =========================================================

DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/"
    "telco-customer-churn-on-icp4d/master/data/"
    "Telco-Customer-Churn.csv"
)


@st.cache_data
def load_dataset():

    df = pd.read_csv(
        DATASET_URL
    )

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    return df


df = load_dataset()


# =========================================================
# SQLITE DATABASE
# =========================================================

@st.cache_resource
def create_database(dataframe):

    connection = sqlite3.connect(
        ":memory:",
        check_same_thread=False
    )

    dataframe.to_sql(
        "customers",
        connection,
        if_exists="replace",
        index=False
    )

    return connection


conn = create_database(df)


# =========================================================
# CUSTOMER LOOKUP
# =========================================================

def get_customer_from_db(customer_id):

    query = """
    SELECT *
    FROM customers
    WHERE customerID = ?
    """

    result = pd.read_sql_query(
        query,
        conn,
        params=(customer_id,)
    )

    if result.empty:
        return None

    return result.iloc[0]


def customer_database_lookup(
    customer_id: str
) -> str:

    """
    Retrieve structured customer information
    from the SQL database.
    """

    customer = get_customer_from_db(
        customer_id
    )

    if customer is None:
        return (
            f"Customer {customer_id} "
            f"was not found in the database."
        )

    return customer.to_json()


# =========================================================
# RULE-BASED RISK ANALYSIS
# =========================================================

def analyze_churn_risk(
    customer_id: str
) -> str:

    """
    Analyze customer churn risk using
    predefined business rules.
    """

    customer = get_customer_from_db(
        customer_id
    )

    if customer is None:
        return (
            f"Customer {customer_id} "
            f"was not found in the database."
        )

    score = 0

    if customer["tenure"] <= 6:
        score += 2

    if customer["Contract"] == "Month-to-month":
        score += 2

    if customer["TechSupport"] == "No":
        score += 1

    if customer["OnlineSecurity"] == "No":
        score += 1

    if score >= 4:
        risk = "High"

    elif score >= 2:
        risk = "Medium"

    else:
        risk = "Low"

    return (
        f"Customer {customer_id}: "
        f"Risk Score = {score}, "
        f"Risk Level = {risk}"
    )


# =========================================================
# MACHINE LEARNING
# =========================================================

@st.cache_resource
def train_ml_pipeline(dataframe):

    X = dataframe.drop(
        columns=[
            "Churn",
            "customerID"
        ]
    )

    y = dataframe["Churn"].map(
        {
            "No": 0,
            "Yes": 1
        }
    )

    categorical_columns = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numerical_columns = X.select_dtypes(
        exclude=["object"]
    ).columns.tolist()

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            ),
            (
                "numerical",
                numerical_pipeline,
                numerical_columns
            )
        ]
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42
                )
            )
        ]
    )

    pipeline.fit(
        X,
        y
    )

    return pipeline


ml_pipeline = train_ml_pipeline(df)


def predict_churn_ml(
    customer_id: str
) -> str:

    """
    Predict customer churn probability using
    the trained machine learning pipeline.
    """

    customer = get_customer_from_db(
        customer_id
    )

    if customer is None:
        return (
            f"Customer {customer_id} "
            f"was not found in the database."
        )

    customer_df = customer.to_frame().T

    customer_df = customer_df.drop(
        columns=["Churn"]
    )

    probability = ml_pipeline.predict_proba(
        customer_df
    )[0][1]

    prediction = (
        "Yes"
        if probability >= 0.5
        else "No"
    )

    return (
        f"Customer {customer_id}: "
        f"Churn Probability = {probability:.2%}, "
        f"Prediction = {prediction}"
    )


# =========================================================
# RETENTION RECOMMENDATION
# =========================================================

def recommend_retention_action(
    customer_id: str
) -> str:

    """
    Recommend a retention action based
    on the customer's actual data.
    """

    customer = get_customer_from_db(
        customer_id
    )

    if customer is None:
        return (
            f"Customer {customer_id} "
            f"was not found in the database."
        )

    if customer["Contract"] == "Month-to-month":

        return (
            f"Customer {customer_id}:\n"
            f"Recommended Action: "
            f"Offer a long-term contract with a suitable incentive.\n"
            f"Reason: "
            f"The customer currently has a "
            f"Month-to-month contract."
        )

    elif customer["TechSupport"] == "No":

        return (
            f"Customer {customer_id}:\n"
            f"Recommended Action: "
            f"Offer a technical support package.\n"
            f"Reason: "
            f"The customer currently does not have TechSupport."
        )

    elif customer["OnlineSecurity"] == "No":

        return (
            f"Customer {customer_id}:\n"
            f"Recommended Action: "
            f"Offer an online security package.\n"
            f"Reason: "
            f"The customer currently does not have OnlineSecurity."
        )

    else:

        return (
            f"Customer {customer_id}:\n"
            f"Recommended Action: "
            f"Offer a personalized retention package.\n"
            f"Reason: "
            f"No specific retention trigger was identified "
            f"by the current recommendation rules."
        )


# =========================================================
# RAG KNOWLEDGE BASE
# =========================================================

RETENTION_POLICY = """
Demo Telecom Retention Policy

Policy 1: Month-to-month contracts
Customers with a Month-to-month contract may be offered
a long-term contract with a suitable incentive.
The goal is to encourage longer-term commitment.

Policy 2: Online Security
Customers who do not currently have OnlineSecurity may be
offered an online security package.

Policy 3: Technical Support
Customers who do not currently have TechSupport may be
offered a technical support package.

Policy 4: High-risk customers
Customers identified as High Risk by the rule-based risk
analysis should be prioritized for retention outreach.

Policy 5: Machine learning predictions
ML churn probability is a model estimate and should be used
as one analytical input. It must not be treated as a guarantee
that a customer will churn.

Policy 6: Historical churn status
Historical Churn Status represents the historical label
available in the dataset. It should not be presented as a
guarantee of the customer's future behavior.
"""


@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


@st.cache_resource
def prepare_rag():

    chunks = [
        chunk.strip()
        for chunk in RETENTION_POLICY.split("\n\n")
        if chunk.strip()
    ]

    embeddings = embedding_model.encode(
        chunks,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    return chunks, embeddings


rag_chunks, rag_embeddings = prepare_rag()


def search_retention_policy(
    query,
    top_k=3
):

    if not query or not query.strip():

        return (
            "No search query was provided."
        )

    query_embedding = embedding_model.encode(
        query,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    scores = util.cos_sim(
        query_embedding,
        rag_embeddings
    )[0]

    top_results = scores.topk(
        k=min(
            top_k,
            len(rag_chunks)
        )
    )

    output = (
        "Relevant retention policy information:\n\n"
    )

    for i, (score, index) in enumerate(
        zip(
            top_results.values,
            top_results.indices
        ),
        start=1
    ):

        output += (
            f"[Result {i} | "
            f"Similarity: {float(score):.3f}]\n"
        )

        output += (
            rag_chunks[int(index)]
            + "\n\n"
        )

    return output


def search_company_retention_policy(
    query: str
) -> str:

    """
    Search the company's retention policy
    knowledge base using semantic search.
    """

    return search_retention_policy(
        query=query,
        top_k=3
    )


# =========================================================
# GEMINI SYSTEM INSTRUCTIONS
# =========================================================

SYSTEM_INSTRUCTIONS = """
You are a Customer Retention AI Agent for a telecom company.

Your job is to understand the user's request, select the appropriate
tools, use their results, and provide a clear business-oriented answer.

IMPORTANT RULES:

1. Use tool results as the primary source of truth.

2. Never invent customer information, statistics, historical facts,
acquisition costs, customer lifetime value, discounts, or unsupported
business claims.

3. Clearly distinguish between:

- Historical Churn Status
- Rule-Based Risk Score
- ML Churn Probability
- ML Prediction
- Retention Recommendation
- Reason for Recommendation
- Company Policy Information

4. The Rule-Based Risk Score is calculated using predefined
business rules. It is NOT an ML prediction.

5. ML Churn Probability is a model estimate, not a guarantee.

6. ML Prediction:
- Yes when churn probability >= 50%
- No when churn probability < 50%

7. Never say that a customer will definitely churn.

8. When the user asks for customer information only,
use the database tool.

9. When the user asks for churn risk,
use the risk analysis tool.

10. When the user asks for ML churn probability or prediction,
use the ML prediction tool.

11. When the user asks for a retention recommendation,
use the retention recommendation tool.

12. When the user asks about company retention policy
or internal policy information, use the RAG policy tool.

13. When a question requires both customer data and company policy,
use the relevant customer tools and the RAG policy tool.

14. When comparing two customers, compare their available data,
risk score, ML probability, ML prediction, and recommendations.

15. Answer in the requested language.

16. Keep answers clear, concise, and business-oriented.

17. Never invent statistics, correlations, causal relationships,
industry trends, or unsupported historical patterns.

18. Retention recommendations must be based on the actual result
of the retention recommendation tool.

19. Company policy retrieved through RAG must be clearly identified
as information from the company knowledge base.

20. Do not treat company policy information as customer-specific
information unless customer tools confirm that the policy applies.

21. When information is missing, say:
"This information is not available from the current tools."
"""


# =========================================================
# GEMINI CLIENT
# =========================================================

@st.cache_resource
def create_gemini_client():

    api_key = st.secrets[
        "GEMINI_API_KEY"
    ]

    return genai.Client(
        api_key=api_key
    )


client = create_gemini_client()


# =========================================================
# AGENT TOOLS
# =========================================================

AGENT_TOOLS = [
    customer_database_lookup,
    analyze_churn_risk,
    predict_churn_ml,
    recommend_retention_action,
    search_company_retention_policy
]


# ================
