import torch


def backward(res):
    def backward_tensor_with_grad(res):
        if isinstance(res, torch.Tensor) and res.grad_fn:
            res.sum().backward()
            return True
        return False

    if not backward_tensor_with_grad(res) and isinstance(res, (tuple, list)):
        for x in res:
            if backward_tensor_with_grad(x):
                break
