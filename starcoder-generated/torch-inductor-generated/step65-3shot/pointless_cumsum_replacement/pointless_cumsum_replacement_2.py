
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = None
        b['layout'] = torch.strided
        b['device'] = torch.device('cpu')
        a['dtype'] = torch.float16
        a['layout'] = None
        a['device'] = torch.device(type='cpu')
        a['dtype_to'] = None
        a['dtype_from'] = torch.float16
        b['dtype_to'] = None
        b['dtype_from'] = torch.float16
        t1 = torch.full([2048, 3072], 1.0, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = t1.to(dtype=a['dtype'])
        t3 = torch.cumsum(t2, 1)
        return t3
# Inputs to the model
x1 = torch.randn(2048, 3072, device='cpu')
