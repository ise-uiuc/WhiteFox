
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        if x1 % 2 == 0:
            a = {}
            b['dtype'] = torch.float32
            b['layout'] = torch.strided
            b['device'] = torch.device('cuda:0')
            a['dtype'] = torch.float64
            a['layout'] = torch.strided
            a['device'] = torch.device('cuda:0')
            a['dtype_to'] = torch.float32
            a['dtype_from'] = torch.float64
            b['dtype_to'] = torch.float64
            b['dtype_from'] = torch.float32
            t1 = torch.full([4, 4, 4, 4], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
            t2 = t1.to(dtype=a['dtype'])
        else:
            a = {}
            b['dtype'] = torch.float32
            b['layout'] = torch.strided
            b['device'] = torch.device('cuda:0')
            a['dtype'] = torch.float64
            a['layout'] = torch.strided
            a['device'] = torch.device('cuda:0')
            a['dtype_to'] = torch.float32
            a['dtype_from'] = torch.float64
            b['dtype_to'] = torch.float64
            b['dtype_from'] = torch.float32
            t1 = torch.full([4, 4, 4, 4], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
            t2 = t1.to(dtype=a['dtype'])
        t3 = torch.cumsum(t2, 1)
        return t3
# Inputs to the model
x1 = torch.randn(4, 4, 4, 4, device='cuda:0')
