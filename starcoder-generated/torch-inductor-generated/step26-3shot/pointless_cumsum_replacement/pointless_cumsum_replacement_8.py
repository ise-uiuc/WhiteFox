
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = torch.int8
        b['layout'] = torch.strided
        b['device'] = torch.device('cpu')
        a['dtype'] = torch.bool
        a['layout'] = torch.strided
        a['device'] = torch.device('cpu')
        a['dtype_to'] = torch.uint8
        a['dtype_from'] = torch.bool
        b['dtype_to'] = torch.uint8
        b['dtype_from'] = torch.bool
        t1 = torch.full([128, 1024, 1024, 1], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = t1.to(dtype=a['dtype'])
        t3 = torch.cumsum(t2, 2)
        return t3
# Inputs to the model
x1 = torch.randn(128, 1024, 1024, 1, device='cpu')
