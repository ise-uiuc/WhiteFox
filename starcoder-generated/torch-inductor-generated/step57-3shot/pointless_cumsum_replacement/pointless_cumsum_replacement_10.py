


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = torch.int64
        b['layout'] = torch.strided
        b['device'] = torch.device('cpu')
        a['dtype'] = torch.float32
        a['layout'] = torch.strided
        a['device'] = torch.device('cpu')
        a['dtype_to'] = torch.int64
        a['dtype_from'] = torch.float32
        b['dtype_to'] = torch.int64
        b['dtype_from'] = torch.float32
        t1 = x1.to(dtype=b['dtype'])
        t2 = torch.cumsum(t1, 1)
        t3 = t2.to(dtype=a['dtype'])
        return t3
# Inputs to the model
x1 = torch.randn(7, 101, device='cpu')
