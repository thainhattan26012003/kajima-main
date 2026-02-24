import torch
import numpy as np
from sklearn.metrics import f1_score
from collections import defaultdict


class Evaluator:
    def __init__(
        self, cfg, many_idxs: list[int], med_idxs: list[int], few_idxs: list[int]
    ):
        self.cfg = cfg
        self.many_idxs = many_idxs
        self.med_idxs = med_idxs
        self.few_idxs = few_idxs

        self.reset()

    def reset(self):
        self._y_pred = []
        self._y_true = []
        self._y_conf = []

        self._correct = 0
        self._total = 0

    def update(self, model_output, true_output):
        """
        model_output: Output of model [batch_size, num_classes]
        true_output: True label [batch_size]
        """

        predicted = model_output.max(1)[1]
        conf = torch.softmax(model_output, dim=1).max(1)[0]
        correct = predicted.eq(true_output).float()

        self._correct += int(correct.sum().item())
        self._total += model_output.shape[0]

        self._y_pred.extend(predicted.data.cpu().numpy().tolist())
        self._y_true.extend(true_output.data.cpu().numpy().tolist())
        self._y_conf.extend(conf.data.cpu().numpy().tolist())

    def evaluate(self):
        results = dict()
        accuracy = self._correct / self._total * 100.0
        err = 100.0 - accuracy
        macro_f1 = 100.0 * f1_score(self._y_true, self._y_pred, average="macro")

        results["accuracy"] = accuracy
        results["error"] = err
        results["macro_f1"] = macro_f1

        print(
            "=> result\n"
            f"* total: {self._total:,}\n"
            f"* correct: {self._correct:,}\n"
            f"* accuracy: {accuracy:.1f}%\n"
            f"* error: {err:.1f}%\n"
            f"* macro_f1: {macro_f1:.1f}%"
        )

        self._per_class_res = defaultdict(list)

        for label, pred in zip(self._y_true, self._y_pred):
            matches = label == pred
            self._per_class_res[label].append(matches)

        print(self._per_class_res)

        labels = sorted(self._per_class_res.keys())
        cls_acc = list()
        for label in labels:
            acc = sum(self._per_class_res[label]) / len(self._per_class_res[label])
            cls_acc.append(acc)
        cls_acc_str = np.array2string(np.array(cls_acc), precision=2)
        print(f"Class accuracy: {cls_acc_str}")

        worst_acc_cls = min(cls_acc)
        # Harmonic mean of per-class accuracies (as ratio in [0, 1])
        harmonic_mean = len(cls_acc) / np.sum(
            [1.0 / max(acc, 0.001) for acc in cls_acc]
        )

        results["worst_case_acc"] = worst_acc_cls
        results["hmean_acc"] = harmonic_mean

        print(
            f"* worst_case_acc: {worst_acc_cls * 100:.1f}%\n"
            f"* hmean_acc: {harmonic_mean * 100:.1f}%\n"
        )

        if (
            self.many_idxs is not None
            and self.med_idxs is not None
            and self.few_idxs is not None
        ):
            many_acc = np.mean(np.array(cls_acc)[self.many_idxs])
            med_acc = np.mean(np.array(cls_acc)[self.med_idxs])
            few_acc = np.mean(np.array(cls_acc)[self.few_idxs])

            results["many_acc"] = many_acc
            results["med_acc"] = med_acc
            results["few_acc"] = few_acc

        print(
            f"* Many acc: {many_acc}", f"* Med acc: {med_acc}", f"* Few acc: {few_acc}"
        )

        mean_acc = np.mean(cls_acc)
        results["mean_acc"] = mean_acc
        print(f"* Mean acc: {mean_acc}")

        return results
