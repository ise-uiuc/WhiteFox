
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = torch.complex128
        b['layout'] = torch.strided
        b['device'] = torch.device('cuda:0')
        a['dtype'] = torch.float32
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:0')
        a['dtype_to'] = torch.complex128
        a['dtype_from'] = torch.float32
        b['dtype_to'] = torch.bool
        b['dtype_from'] = torch.complex128
        t1 = torch.full([256], 1+1j, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = t1.to(dtype=a['dtype'])
        t3 = torch.isnan(t2)
        return t3
# Inputs to the model
x1 = torch.randn(256, device='cuda:0')
