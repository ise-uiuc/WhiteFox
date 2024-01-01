
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = torch.uint16
        b['layout'] = torch.strided
        b['device'] = torch.device('cuda:1')
        a['dtype'] = torch.int8
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:0')
        a['dtype_to'] = torch.uint16
        a['dtype_from'] = torch.bool
        b['dtype_to'] = torch.uint16
        b['dtype_from'] = torch.bool
        t1 = torch.full([792, 2560], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = t1.to(dtype=a['dtype'])
        t3 = torch.cumsum(t2, 1)
        return t3
# Inputs to the model
x1 = torch.randn(792, 2560, device='cuda:0')
