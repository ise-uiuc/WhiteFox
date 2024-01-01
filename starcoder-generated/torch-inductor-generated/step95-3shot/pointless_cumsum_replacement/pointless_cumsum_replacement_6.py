
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = torch.bool
        b['layout'] = torch.strided
        a['dtype'] = torch.int16
        a['layout'] = torch.strided
        a['device'] = torch.device('cpu')
        a['dtype_to'] = torch.float16
        a['dtype_from'] = torch.int64
        b['dtype_to'] = torch.bool
        b['dtype_from'] = torch.float16
        b['device'] = torch.device('cpu')
        t1 = torch.full([256, 64], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = t1.to(dtype=a['dtype'])
        t3 = torch.cumsum(t2, 1)
        return t3
# Inputs to the model
x1 = torch.randn(256, 64, dtype=torch.int16, device='cpu')
