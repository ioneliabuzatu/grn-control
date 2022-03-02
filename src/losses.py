def jax_bce_w_logits(x, y, weight=None, average=True):
    """
    Binary Cross Entropy Loss
    Should be numerically stable, built based on: https://github.com/pytorch/pytorch/issues/751
    :param x: Input tensor
    :param y: Target tensor
    :param weight: Vector of example weights
    :param average: Boolean to average resulting loss vector
    :return: Scalar value
    """
    max_val = jnp.clip(x, 0, None)
    loss = x - x * y + max_val + jnp.log(jnp.exp(-max_val) + jnp.exp((-x - max_val)))

    if weight is not None:
        loss = loss * weight

    if average:
        loss_mean = loss.mean()
        assert loss_mean > 0
        return loss_mean
    else:
        loss_sum = loss.sum()
        assert loss_sum > 0
        return loss_sum
