# eval_retriever.py
from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric

from src.retriever import build_retriever
from src.reranker import RerankingRetriever
from evals.harness import load_goldens, summarize_by_metric, print_summary

load_dotenv()

GOLDEN_PATH = "goldens/retriever_goldens.json"
RETRIEVER_NAME = "reranker"   # which retriever to run; override from the CLI
JUDGE_MODEL = "gpt-4o-mini"   # NOTE: differs from the other evals (gpt-4o-mini)
TOP_K = 5
THRESHOLD = 0.7

# 0. THE CONTENDERS --- the name picks the retriever AND tags the report, so the
#    two can never drift apart. Both hand back TOP_K docs, so the only thing
#    that differs between runs is the ranking.
RETRIEVERS = {
    "baseline": lambda: build_retriever(TOP_K),
    "reranker": lambda: RerankingRetriever(fetch_k=2 * TOP_K, top_k=TOP_K),
}


def run(retriever, name=RETRIEVER_NAME):
    # 1. LOAD the golden set --- the fixed, human-authored truth
    goldens = load_goldens(GOLDEN_PATH)

    # 2. RUN THE INJECTED RETRIEVER on each question to fill retrieval_context,
    #    then build one test case per golden.
    test_cases = []
    for g in goldens:
        retrieved = retriever.invoke(g["query"])
        retrieval_context = [doc.page_content for doc in retrieved]

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                expected_output=g["ideal_answer"],
                retrieval_context=retrieval_context,
                actual_output="(generator not evaluated in this run)",
            )
        )

    # 3. THE METRICS --- recall (did we miss?) and precision (ranked well?)
    metrics = [
        ContextualRecallMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
        ContextualPrecisionMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    ]

    # 4. EVALUATE --- every metric on every case, batched + parallel, printed report.
    #    hyperparameters travel with the run so the report is tagged with the config.
    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        hyperparameters={
            "retriever": name,               # whichever one was actually injected
            "embedding_model": "text-embedding-3-large",
            "chunk_size": 1000,
            "chunk_overlap": 150,
            "top_k": TOP_K,
            "judge_model": JUDGE_MODEL,
            "golden_set": GOLDEN_PATH,
        },
    )
    return summarize_by_metric(result)


def run_local(name=RETRIEVER_NAME):
    """Standalone convenience: build the named retriever, then run."""
    if name not in RETRIEVERS:
        raise SystemExit(
            f"unknown retriever {name!r}; pick one of: {', '.join(RETRIEVERS)}"
        )
    return run(RETRIEVERS[name](), name=name)


# python -m evals.eval_retriever [baseline|reranker]
if __name__ == "__main__":
    import sys

    which = sys.argv[1] if len(sys.argv) > 1 else RETRIEVER_NAME
    print_summary(f"retriever ({which})", run_local(which))