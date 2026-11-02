from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


def compute_text_metrics(preds, targets):
    """
    preds: list[str]
    targets: list[str]
    """

    bleu_scores = []
    rouge_scores = []

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    smooth = SmoothingFunction().method1

    for pred, tgt in zip(preds, targets):
        pred = pred.strip().lower()
        tgt = tgt.strip().lower()

        if len(pred) == 0 or len(tgt) == 0:
            bleu_scores.append(0.0)
            rouge_scores.append(0.0)
            continue

        bleu = sentence_bleu(
            [tgt.split()],
            pred.split(),
            smoothing_function=smooth
        )
        bleu_scores.append(bleu)

        rouge = scorer.score(tgt, pred)['rougeL'].fmeasure
        rouge_scores.append(rouge)

    return {
        "bleu": sum(bleu_scores) / max(len(bleu_scores), 1),
        "rougeL": sum(rouge_scores) / max(len(rouge_scores), 1)
    }