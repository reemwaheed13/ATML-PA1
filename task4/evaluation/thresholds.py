import numpy as np

from task4.evaluation.metrics import auroc


# tau = 95th percentile of unknownness on CIFAR-10 val -> accepts ~95% of known
def calibrate(u_val, tpr=0.95):
    return float(np.quantile(u_val, tpr))


# accept iff u <= tau
def accept_rate(u, tau):
    return float(np.mean(u <= tau))


def reject_rate(u, tau):
    return float(np.mean(u > tau))


# aurocs + validation-calibrated acceptance/rejection for one score
def osr_row(u_val, u_test, u_near, u_far):
    tau = calibrate(u_val, 0.95)
    all_unknown = np.concatenate([u_near, u_far])
    return {
        'auroc_near': auroc(u_test, u_near),
        'auroc_far':  auroc(u_test, u_far),
        'auroc_all':  auroc(u_test, all_unknown),
        'tau': tau,
        'test_accept': accept_rate(u_test, tau),
        'near_reject': reject_rate(u_near, tau),
        'far_reject':  reject_rate(u_far, tau),
        'fpr95_near':  accept_rate(u_near, tau),  # unknowns wrongly accepted
        'fpr95_far':   accept_rate(u_far, tau),
    }
