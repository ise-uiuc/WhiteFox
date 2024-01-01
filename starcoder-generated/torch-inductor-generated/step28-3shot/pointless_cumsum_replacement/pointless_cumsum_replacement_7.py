
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = torch.bool
        b['layout'] = torch.strided
        b['device'] = torch.device('cpu')
        a['dtype'] = torch.float32
        a['layout'] = torch.sparse_coo
        a['device'] = torch.device('cpu')
        a['dtype_to'] = torch.float32
        a['dtype_from'] = torch.bool
        b['dtype_to'] = torch.bool
        b['dtype_from'] = torch.float32
        t1 = torch.full([1, 100], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = t1.to(dtype=a['dtype'], layout=a['layout'])
        t3 = torch.cumsum(t2, 1)
        return t3
# Inputs to the model
x1 = torch.randn(100, 1, device='cpu')
