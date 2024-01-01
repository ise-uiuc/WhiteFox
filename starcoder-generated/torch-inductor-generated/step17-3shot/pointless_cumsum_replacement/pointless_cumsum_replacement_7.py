
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        c = {}
        a = {}
        b['dtype'] = torch.complex128
        b['layout'] = torch.sparse_coo
        b['device'] = torch.device('cpu')
        c['dtype'] = torch.complex128
        c['layout'] = torch.strided
        c['device'] = torch.device('cpu')
        c['dtype_to'] = torch.complex128
        c['dtype_from'] = torch.complex64
        b['dtype_to'] = torch.complex64
        b['dtype_from'] = torch.complex128
        a['dtype'] = torch.float32
        a['layout'] = torch.strided
        a['device'] = torch.device('cpu')
        a['dtype_to'] = torch.float32
        a['dtype_from'] = torch.complex64
        t1 = torch.full([1, 3, 9, 1], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = t1.to(dtype=c['dtype'])
        t3 = torch.addmm(t2, t2, t2)
        t4 = t3.to(dtype=a['dtype'])
        t5 = torch.addmm(t1, t1, t4)
        return t5
# Inputs to the model
x1 = torch.randn(1, 3, 9, 1, device='cpu')
