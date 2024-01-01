
class Model(torch.nn.Module):
    pass
    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = torch.complex128
        b['layout'] = torch.strided
        b['device'] = torch.device('cuda:0')
        a['dtype'] = torch.double
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:0')
        a['dtype_to'] = torch.complex128
        a['dtype_from'] = torch.double
        b['dtype_to'] = torch.complex128
        b['dtype_from'] = torch.double
        t1 = torch.full([5, 6, 7], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = t1.to(dtype=a['dtype'])
        t3 = torch.cumsum(t2, 1)
        t4 = t3.to(dtype=b['dtype'])
        return t4
# Inputs to the model
x1 = torch.randn(5, 6, 7, device='cuda:0')
