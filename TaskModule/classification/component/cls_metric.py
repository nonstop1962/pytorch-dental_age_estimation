import logging

import numpy as np

logger = logging.getLogger("Logger")


def cls_metric(cfg_metric):
    metric = _get_metric_instance(cfg_metric["name"])
    metric_name = cfg_metric.pop("name")
    metric = metric(**cfg_metric)

    logger.info(f'[{"METRIC".center(9)}] {metric_name} [params] {cfg_metric}')

    return metric


class runningScore_cls(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask],
            minlength=n_class ** 2,
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(
                lt.flatten(), lp.flatten(), self.n_classes
            )

    def get_scores(self):
        """Returns accuracy score evaluation result.
        - overall accuracy
        - mean accuracy
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)

        return {
            "Overall Acc": acc,
            "Mean Acc": acc_cls,
            "Confusion Mat \n": hist,
        }

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


def _get_metric_instance(name):
    try:
        return {"runningScore_cls": runningScore_cls}[name]
    except:
        raise (f"Model {name} not available")
