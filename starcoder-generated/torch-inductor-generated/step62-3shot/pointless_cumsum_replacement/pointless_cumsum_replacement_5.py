
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        b_ = {}
        a_ = {}
        b_['dtype'] = x2.dtype
        b_['layout'] = torch.strided
        b_['device'] = x2.device
        a_['dtype'] = x1.dtype
        a_['layout'] = torch.strided
        a_['device'] = x1.device
        a_['dtype_to'] = torch.float32
        a_['dtype_from'] = x1.dtype
        b_['dtype_to'] = torch.float32
        b_['dtype_from'] = x2.dtype
        t1 = torch.full([16384, 262144], 1, dtype=b_['dtype'], layout=b_['layout'], device=b_['device'], pin_memory=False)
        t2 = t1.to(dtype=a_['dtype'])
        t3 = torch.cumsum(t2, 1)
        return t3
# Inputs to the model
x1 = torch.randn(16384, 262144, device='cuda:0')
x2 = torch.randn(16384, 262144, device='cuda:0')
