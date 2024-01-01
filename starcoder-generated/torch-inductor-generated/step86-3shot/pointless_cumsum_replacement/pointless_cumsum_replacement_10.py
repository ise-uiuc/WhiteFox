
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        c = {}
        b = {}
        a = {}
        a['dtype'] = torch.int8
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:0')
        a['dtype_to'] = torch.int8
        a['dtype_from'] = torch.float64
        c['dtype_to'] = torch.uint8
        c['dtype_from'] = torch.int8
        t1 = torch.cumsum(torch.full([x1, x1], 1, dtype=a['dtype'], layout=a['layout'], device=a['device'], pin_memory=False), 0)
        t2 = t1.to(dtype=c['dtype'])
        t3 = torch.cumsum(t2, 1)
        b['dtype'] = torch.float64
        b['layout'] = torch.strided
        b['device'] = torch.device('cuda:0')
        t4 = t3.to(dtype=b['dtype'])
        return t4
# Inputs to the model
x1 = 1
