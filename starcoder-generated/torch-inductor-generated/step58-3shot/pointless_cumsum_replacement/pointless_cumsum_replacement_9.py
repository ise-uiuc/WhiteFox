
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        b['dtype'] = torch.float
        b['layout'] = torch.strided
        b['device'] = torch.device('cuda:0')
        b['dtype_to'] = torch.double
        b['dtype_from'] = torch.float64
        t1 = torch.full([7, 2490880], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = t1.to(dtype=b['dtype_to'])
        t3 = torch.cumsum(t2, 1)
        return t3
# Inputs to the model
x1 = torch.randn(7, 2490880, device='cuda:0')
