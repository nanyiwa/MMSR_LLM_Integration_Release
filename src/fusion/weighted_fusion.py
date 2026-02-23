# 简单加权融合，由于混合融合尝试结果不佳，最终决定仍然采用简单加权融合
class WeightedFusion:
    def __init__(self):
        # 定义标准标签顺序，确保所有模态对齐
        self.emotions = ['happy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'neutral']

    def fuse(self, audio_scores, text_scores, vision_scores, weights=None):
        """
        简单加权融合
        :param audio_scores: dict {'happy': 0.1, ...} or None
        :param text_scores: dict or None
        :param vision_scores: dict or None
        :param weights: dict {'audio': 0.4, 'text': 0.3, 'vision': 0.3}
        :return: 融合后的 dict
        """
        if weights is None:
            # 默认权重
            weights = {'audio': 0.3, 'text': 0.4, 'vision': 0.3}

        # 1. 初始化最终得分
        final_scores = {emo: 0.0 for emo in self.emotions}

        # 2. 检查有效模态并归一化权重
        # 如果用户只传了音频，不让文本和视觉的缺失导致总分变低
        active_weights = 0.0
        if audio_scores: active_weights += weights.get('audio', 0)
        if text_scores:  active_weights += weights.get('text', 0)
        if vision_scores: active_weights += weights.get('vision', 0)

        if active_weights == 0:
            return final_scores

        # 3. 加权累加
        for emo in self.emotions:
            score = 0.0
            if audio_scores:
                score += audio_scores.get(emo, 0) * weights.get('audio', 0)
            if text_scores:
                score += text_scores.get(emo, 0) * weights.get('text', 0)
            if vision_scores:
                score += vision_scores.get(emo, 0) * weights.get('vision', 0)

            # 重新归一化
            final_scores[emo] = score / active_weights

        # 4. 排序输出
        sorted_scores = dict(sorted(final_scores.items(), key=lambda x: x[1], reverse=True))
        return sorted_scores