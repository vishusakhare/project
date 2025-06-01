# app_gradio.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
from evaluation.grammar import grammar_check
from evaluation.coherence import coherence_score
from evaluation.factual import factual_accuracy
# If the actual function in style.py is named `evaluate_style` for example:
from evaluation.style import style_score

from evaluation.relevance import relevance_score
from helpers.scoring import aggregate_scores
 

def evaluate_text(user_text, context, reference_facts):
    weights = {
        "grammar": 0.2,
        "coherence": 0.2,
        "factual": 0.3,
        "style": 0.1,
        "relevance": 0.2
    }

    grammar = grammar_check(user_text)
    coherence = coherence_score(user_text)
    factual = factual_accuracy(user_text, reference_facts)
    style = style_score(user_text, "high school students")
    relevance = relevance_score(user_text, context)

    scores = {
        "grammar": grammar["score"],
        "coherence": coherence["score"],
        "factual": factual["score"],
        "style": style["score"],
        "relevance": relevance["score"]
    }

    overall = aggregate_scores(scores, weights)

    result = f"""
    📘 **Grammar Score:** {grammar['score']}/100  
    Suggestions: {grammar['suggestions']}

    🧠 **Coherence Score:** {coherence['score']}/100  
    Comment: {coherence['comment']}

    📚 **Factual Accuracy:** {factual['score']}/100  
    Comment: {factual['comment']}

    ✍️ **Style Score:** {style['score']}/100  
    Suggestion: {style['suggestion']}

    🎯 **Relevance Score:** {relevance['score']}/100  
    Audience Fit: {relevance['audience_fit']}/100  
    Purpose Alignment: {relevance['purpose_alignment']}/100  
    Context Adherence: {relevance['context_adherence']}/100  
    Comment: {relevance['comment']}

    🏁 **Composite Score:** {overall}/100
    """
    return result

iface = gr.Interface(
    fn=evaluate_text,
    inputs=[
        gr.Textbox(lines=10, label="AI-Generated Text"),
        gr.Textbox(lines=3, label="Context (Audience, Purpose, etc.)"),
        gr.Textbox(lines=3, label="Reference Facts (Optional)")
    ],
    outputs="markdown",
    title="TextEval: Automated Text Evaluation (Gradio Edition)",
    description="Evaluates AI-generated text for grammar, coherence, factual accuracy, style, and contextual relevance."
)

if __name__ == "__main__":
    iface.launch()
