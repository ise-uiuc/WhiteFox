
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        c = {}
        d = {}
        b['dtype'] = torch.float64
        b['layout'] = torch.strided
        b['device'] = torch.device('cuda:0')
        a['dtype'] = torch.float16
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:0')
        a['dtype_to'] = torch.float64
        a['dtype_from'] = torch.half
        d['dtype'] = torch.float64
        d['layout'] = torch.strided
        d['device'] = torch.device('cuda:1')
        d['dtype_to'] = torch.float64
        d['dtype_from'] = torch.float16
        c['dtype'] = torch.float64
        c['layout'] = torch.strided
        c['device'] = torch.device('cuda:1')
        c['dtype_to'] = torch.float64
        c['dtype_from'] = torch.float64
        b['torch2trt_enabled'] = False
        t1 = torch.full([1, 1], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = t1.to(dtype=a['dtype'])
        t3 = t2.to(dtype=c['dtype'])
        t4 = t3.to(dtype=d['dtype'])
        return t4
# Inputs to the model
x1 = torch.randn(1, 1, device='cuda:0')
