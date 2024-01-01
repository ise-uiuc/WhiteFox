
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = torch.char
        b['layout'] = torch.strided
        b['device'] = torch.device('cpu')
        a['dtype'] = torch.long
        a['layout'] = torch.strided
        a['device'] = torch.device('cpu')
        a['dtype_to'] = torch.bool
        a['dtype_from'] = torch.long
        b['dtype_to'] = torch.bool
        b['dtype_from'] = torch.char
        t1 = torch.full([4, 416], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = t1.to(dtype=a['dtype'])
        t3 = torch.cumsum(t2, 1)
        return t3
# Inputs to the model
x1 = torch.randn(4, 416, dtype=torch.bool, device='cpu')
