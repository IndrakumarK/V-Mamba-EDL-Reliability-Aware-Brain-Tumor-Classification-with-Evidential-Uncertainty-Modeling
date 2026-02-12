from .evidential_loss import evidential_loss
from .prototype_loss import prototype_loss
from .kl_regularization import kl_divergence


def total_loss(alpha, target_onehot, similarity, labels,
               lambda1=0.3, lambda2=0.2):
    loss_e = evidential_loss(alpha, target_onehot)
    loss_p = prototype_loss(similarity, labels)
    loss_kl = kl_divergence(alpha, alpha.shape[1])
    return loss_e + lambda1 * loss_p + lambda2 * loss_kl
