
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3):
        b = {}
        a = {}
        c = {}
        b['dtype'] = torch.float32
        b['layout'] = torch.strided
        b['device'] = torch.device('cuda:0')
        a['dtype'] = torch.float64
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:0')
        c['dtype'] = torch.float64
        c['layout'] = torch.strided
        c['device'] = torch.device('cuda:0')
        a['dtype_to'] = torch.float32
        a['dtype_from'] = torch.float64
        b['dtype_to'] = torch.float32
        b['dtype_from'] = torch.float64
        c['dtype_to'] = torch.float32
        c['dtype_from'] = torch.float64
        t1 = x1.to(dtype=b['dtype'])
        t2 = torch.full([4096, 4096], 1, dtype=c['dtype'], layout=c['layout'], device=c['device'], pin_memory=False)
        t3 = torch.full([4096, 4096], 1, dtype=c['dtype'], layout=c['layout'], device=c['device'], pin_memory=False)
        t1 = t2.to(dtype=c['dtype'])
        t1 = t3 - t2
        t1 = t1.to(dtype=b['dtype'])
        return t1
# Inputs to the model
x1 = torch.randn(4096, 4096, device='cuda:0')
x2 = torch.randn(4096, 4096, device='cuda:0')
x3 = torch.randn(4096, 4096, device='cuda:0')
