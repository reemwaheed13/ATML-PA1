import torch


def different_class_partner(y):
    # for each i, pick a partner index j (uniformly) with y[j] != y[i]
    n = len(y)
    partner = torch.empty(n, dtype=torch.long, device=y.device)
    for i in range(n):
        cand = (y != y[i]).nonzero(as_tuple=True)[0]
        if len(cand) == 0:
            partner[i] = i  # degenerate: whole half is one class
        else:
            partner[i] = cand[torch.randint(len(cand), (1,), device=y.device)]
    return partner


def manifold_mixup(net, x, y, dist):
    # mix representations after layer2 (forward_pre), then finish through forward_post.
    # one lambda ~ Beta per call; partners are guaranteed a different class.
    h = net.forward_pre(x)
    partner = different_class_partner(y)
    lam = dist.sample().to(h.device)
    h_mix = lam * h + (1.0 - lam) * h[partner]
    return net.forward_post(h_mix)
