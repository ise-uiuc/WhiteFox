
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        b = {}
        c = {}
        a = {}
        b['dtype'] = torch.float32
        b['layout'] = torch.strided
        b['device'] = torch.device('cuda:0')
        c['dtype'] = torch.float64
        c['layout'] = torch.strided
        c['device'] = torch.device('cpu:0')
        a['dtype'] = torch.float16
        a['layout'] = torch.strided
        a['device'] = torch.device('cpu:0')
        a['dtype_to'] = torch.float64
        a['dtype_from'] = torch.float32
        b['dtype_to'] = torch.double
        b['dtype_from'] = torch.float64
        t1 = torch.full([33, 3], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = x1.to(dtype=a['dtype'])
        t3 = torch.cumsum(t2, 1)
        t4 = t3.to(dtype=c['dtype'])
        return t4
# Inputs to the model
x1 = torch.randn(33, 3, device='cuda:0')
x2 = 1
