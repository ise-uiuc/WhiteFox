
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        a = {}
        a['dtype'] = torch.int16
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:0')
        b = {}
        b['dtype'] = torch.float64
        b['layout'] = torch.strided
        b['device'] = torch.device('cuda:0')
        a['dtype_to'] = torch.float64
        a['dtype_from'] = torch.int16
        b['dtype_to'] = torch.float64
        b['dtype_from'] = torch.int16
        t1 = torch.full([4, 4], 1, dtype=a['dtype'], layout=a['layout'], device=a['device'], pin_memory=False)
        t2 = t1.to(dtype=b['dtype'])
        t3 = torch.cumsum(t2, 1)
        return t3
# Inputs to the model
x1 = torch.randn(4, 4, device='cuda:0')
