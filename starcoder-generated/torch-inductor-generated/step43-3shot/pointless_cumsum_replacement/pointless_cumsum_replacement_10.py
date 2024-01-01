
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        b['dtype'] = torch.float32
        b['layout'] = torch.strided
        b['device'] = torch.device('cuda:0')
        b['dtype_to'] = torch.float64
        b['dtype_from'] = torch.float
        t1 = torch.full([3136, 872], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = t1.to(dtype=b['dtype_to'])
        t3 = torch.cumsum(t2, 1)
        return t3
# Inputs to the model
x1 = torch.randn(3136, 872, device='cuda:0')
