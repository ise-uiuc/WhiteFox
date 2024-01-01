
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = torch.float64
        b['layout'] = torch.strided
        b['device'] = torch.device('cpu')
        a['dtype_to'] = torch.float64
        b['dtype_to'] = torch.float64
        t1 = torch.full([1, 384], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = torch.cumsum(t1, 1)
        return t2
# Inputs to the model
x1 = torch.randn(1, 384, device='cpu')
