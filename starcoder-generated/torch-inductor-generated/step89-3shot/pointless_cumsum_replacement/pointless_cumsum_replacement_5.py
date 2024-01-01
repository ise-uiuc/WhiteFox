
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = torch.float
        b['layout'] = torch.strided
        b['device'] = torch.device('cuda:0')
        a['dtype'] = torch.float
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:0')
        a['dtype_to'] = torch.float
        a['dtype_from'] = torch.float
        b['dtype_to'] = torch.float
        b['dtype_from'] = torch.float
        t1 = torch.full([384, 2048], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = t1.to(dtype=a['dtype'])
        t3 = torch.cumsum(t2, 0)
        return t3
# Inputs to the model
x1 = torch.randn(384, 2048, device='cuda:0')
