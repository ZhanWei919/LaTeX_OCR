import numpy as np
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import editdistance


def evaluate(losses, top3accs, references, hypotheses, logger=None):
    """
    评估模型性能，计算以下指标:
    - BLEU-4
    - TOP-3 ACCURACY
    - Edit Distance
    - LOSS
    并计算综合评分 Score = (BLEU-4 + Edit Distance) / 2
    """
    smoothing = SmoothingFunction().method1

    # references 现在的格式是 [[ref1], [ref2], ...] 其中每个refN是一个token列表
    # hypotheses 现在的格式是 [hyp1, hyp2, ...] 其中每个hypN是一个token列表
    bleu4 = corpus_bleu(references, hypotheses,
                        weights=(0.25, 0.25, 0.25, 0.25),
                        smoothing_function=smoothing)

    # 获取 LOSS 和 TOP-3 ACCURACY
    loss = losses.avg
    top3acc = top3accs.avg

    # 计算编辑距离
    total_distance = 0
    for ref, hyp in zip(references, hypotheses):
        ref_str = ' '.join(map(str, ref[0]))  # ref[0]因为每个参考是单元素列表
        hyp_str = ' '.join(map(str, hyp))
        edit_dist = editdistance.eval(ref_str, hyp_str)
        normalized_distance = 1 - (edit_dist / max(len(ref_str), len(hyp_str)))
        total_distance += normalized_distance
    edit_distance = total_distance / len(references) if references else 0

    # 计算综合评分
    score = (bleu4 + edit_distance) / 2

    # 生成评估结果字符串
    eval_result = (
        f"LOSS: {loss:.3f}, "
        f"TOP-3 ACCURACY: {top3acc:.3f}, "
        f"BLEU-4: {bleu4:.3f}, "
        f"Edit Distance: {edit_distance:.3f}, "
        f"Score: {score:.6f}"
    )

    if logger:
        logger.info(eval_result)
    else:
        print(eval_result)

    # 返回综合评分作为主要评估指标
    return score


def exact_match_score(references, hypotheses):
    exact_match = 0
    for ref, hypo in zip(references, hypotheses):
        if np.array_equal(ref, hypo):
            exact_match += 1

    return exact_match / float(max(len(hypotheses), 1))


def edit_distance(references, hypotheses):
    d_leven, len_tot = 0, 0
    for ref, hypo in zip(references, hypotheses):
        d_leven += distance.levenshtein(ref, hypo)
        len_tot += float(max(len(ref), len(hypo)))

    return 1. - d_leven / len_tot
