
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        t = {}
        b['dtype'] = torch.float64
        b['layout'] = torch.strided
        b['device'] = torch.device('cuda:0')
        t['dtype'] = torch.float64
        t['layout'] = torch.strided
        t['device'] = torch.device('cuda:0')
        a['dtype'] = torch.float64
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:0')
        a['dtype_to'] = torch.float64
        a['dtype_from'] = torch.float64
        b['dtype_to'] = torch.float64
        b['dtype_from'] = torch.float64
        t1 = torch.full([512, 2048], 1, dtype=t['dtype'], layout=t['layout'], device=t['device'], pin_memory=False)
        t2 = t1.to(dtype=b['dtype'])
        t3 = t2.to(dtype=a['dtype'])
        t4 = torch.cumsum(t3, 1)
        return t4
# Inputs to the model
x1 = torch.randn(512, 2048, device='cuda:0')
