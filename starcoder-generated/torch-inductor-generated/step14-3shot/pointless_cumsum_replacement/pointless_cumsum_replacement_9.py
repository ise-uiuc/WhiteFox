
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        c = {}
        b['dtype'] = torch.float32
        b['layout'] = torch.strided
        b['device'] = torch.device('cuda:0')
        a['dtype'] = torch.bool
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:0')
        a['dtype_to'] = torch.uint8
        a['dtype_from'] = torch.bool
        b['dtype_to'] = torch.uint8
        b['dtype_from'] = torch.float32
        c['dtype'] = torch.float64
        c['layout'] = torch.strided
        c['device'] = torch.device('cuda:0')
        c['dtype_to'] = torch.float32
        c['dtype_from'] = torch.float64
        t1 = torch.full([1024, 2], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = t1.to(dtype=a['dtype'])
        t3 = t2.boolmask()
        t4 = torch.cumsum(t3, 1)
        t5 = t4.boolmask()
        t6 = t5.to(dtype=c['dtype'])
        t7 = torch.cumsum(t6, 1)
        return t7
# Inputs to the model
x1 = torch.randn(1024, 2, device='cuda:0')
