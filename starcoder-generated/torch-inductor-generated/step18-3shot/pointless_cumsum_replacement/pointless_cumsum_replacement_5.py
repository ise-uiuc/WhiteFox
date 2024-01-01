
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        a['dtype'] = torch.float64
        a['layout'] = torch.strided
        a['device'] = torch.device('cpu')
        b['dtype'] = torch.float32
        b['layout'] = torch.strided
        b['device'] = torch.device('cpu')
        a['dtype_to'] = torch.float32
        a['dtype_from'] = torch.float64
        b['dtype_to'] = torch.float32
        b['dtype_from'] = torch.float32
        t1 = torch.full([1, 512], 1.0, dtype=a['dtype'], layout=a['layout'], device=a['device'], pin_memory=False)
        t2 = t1.to(dtype=b['dtype'])
        t3 = torch.cumsum(t2, 0)
        return t3
# Inputs to the model
x1 = torch.tensor([0.396, 1.678, -2.5152, -0.5695, -0.1087], device='cpu')
