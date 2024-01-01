
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = torch.uint8
        b['layout'] = torch.strided
        b['device'] = torch.device('cuda:0')
        a['dtype'] = torch.float64
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:0')
        a['dtype_to'] = torch.complex128
        a['dtype_from'] = torch.float16
        b['dtype_to'] = torch.float64
        b['dtype_from'] = torch.complex128
        t1 = torch.full([1024, 128], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t11 = t1.to(device='cuda:1')
        t2 = None
        t3 = None
        t33 = torch.bmm(t1, t2.t())
        t4 = t33.to(dtype=a['dtype'])
        t5 = torch.cumsum(t4, 1)
        return t5
# Inputs to the model
x1 = torch.randn(1024, 128, device='cuda:0')
