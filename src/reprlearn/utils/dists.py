import torch
def kld_std(mu1: torch.Tensor, cov1: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    """Assume mu1 and cov1 are mini-batch of multi-dimen vectors.
    - mu1 (torch.tensor): shape is (bs, dim_z)
    - cov1 (torch.tensor): same shape
    """
    assert mu1.shape == cov1.shape
    assert reduction in ["mean", "sum"]
    d = mu1.shape[-1]
    dtype = cov1.dtype
    mu1_norms = torch.sum(mu1.pow(2), dim=1) # torch.linalg.norm(mu1, dim=1)  # (bs,)
    batch_traces = torch.sum(cov1, dim=1)  # (bs,)

    # Compute torch.prod in torch.float64 precision
    batch_logdets = torch.prod(cov1.to(torch.float64), dim=1).log()  # (bs,)
    batch_logdets = batch_logdets.to(dtype)
    if batch_logdets.isinf().any() or batch_logdets.isnan().any():
        breakpoint()
    batch_klds = .5 * (mu1_norms + batch_traces - batch_logdets - d)  # a list of kld values; Don't ignore "-k" term!

    if reduction == 'mean':
        return batch_klds.mean()
    else:
        return batch_klds.sum()

    # vs.         kld = torch.mean(0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
