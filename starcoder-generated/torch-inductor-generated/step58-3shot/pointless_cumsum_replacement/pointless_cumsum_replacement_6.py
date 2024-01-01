
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = torch.int32
        b['layout'] = torch.strided
        a['dtype'] = torch.cfloat
        b['device'] = torch.device('cuda:0')
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:0')
        a['dtype_to'] = torch.complex128
        a['dtype_from'] = torch.cfloat
        b['dtype_to'] = torch.complex128
        b['dtype_from'] = torch.cfloat
        t1 = torch.full([512, 1], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = t1.to(dtype=a['dtype'])
        t3 = torch.cumsum(t2, 1)
        return t3
# Inputs to the model
x1 = torch.randn(512, 1, device='cuda:0')
