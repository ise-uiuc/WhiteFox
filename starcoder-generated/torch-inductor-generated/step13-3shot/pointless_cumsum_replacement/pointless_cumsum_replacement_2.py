
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = torch.complex128
        b['layout'] = torch.strided
        b['device'] = torch.device('cuda:0')
        a['dtype'] = torch.complex32
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:0')
        a['dtype_to'] = torch.complex32
        a['dtype_from'] = torch.complex128
        b['dtype_to'] = torch.complex128
        b['dtype_from'] = torch.complex32
        t1 = torch.full([8, 16], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = t1.to(dtype=a['dtype'])
        t3 = torch.cumsum(t2, 1)
        return t3
# Inputs to the model
x1 = torch.randn(8, 16, 2, device='cuda:0')
