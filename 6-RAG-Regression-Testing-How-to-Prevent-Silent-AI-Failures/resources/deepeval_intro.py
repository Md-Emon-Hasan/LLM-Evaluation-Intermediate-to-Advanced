from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

load_dotenv()

# --- Test case 1: a good answer (should PASS) ---
case_1 = LLMTestCase(
    input="What is the capital of France?",
    actual_output="The capital of France is Paris.",
)

# --- Test case 2: an off-topic answer (should FAIL) ---
case_2 = LLMTestCase(
    input="What is the capital of France?",
    actual_output="France is a beautiful country famous for its food and wine.",
)

# --- One metric, judged by an LLM (pinned for reproducibility) ---
metric = AnswerRelevancyMetric(threshold=0.7, model="gpt-4.1", include_reason=True)

# --- Run BOTH cases through the metric, with a printed report ---
evaluate(test_cases=[case_1, case_2], metrics=[metric])