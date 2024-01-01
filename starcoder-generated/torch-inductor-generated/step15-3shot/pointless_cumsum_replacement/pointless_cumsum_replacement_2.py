
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = torch.bool
        b['layout'] = torch.strided
        b['device'] = torch.device('cuda:0')
        a['dtype'] = torch.int16
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:0')
        a['dtype_to'] = torch.int16
        a['dtype_from'] = torch.int16
        b['dtype_to'] = torch.bool
        b['dtype_from'] = torch.int16
        t1 = torch.full([2048, 1], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = t1.to(dtype=a['dtype'])
        t3 = torch.cumsum(t2, 1)
        t4 = t3.to(dtype=b['dtype'])
        return t4
# Inputs to the model
x1 = torch.randn(2048, 1, device='cuda:0')
