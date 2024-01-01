
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        a['dtype'] = torch.float64
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:0')
        b['dtype'] = torch.float64
        b['layout'] = torch.strided
        b['device'] = torch.device('cuda:0')
        a['dtype_to'] = torch.float64
        a['dtype_from'] = torch.float64
        b['dtype_to'] = torch.float64
        b['dtype_from'] = torch.float64
        t1 = torch.full([2, 3], 1.0, dtype=a['dtype'], layout=a['layout'], device=a['device'], pin_memory=False)
        t2 = t1.to(dtype=b['dtype'])
        t3 = torch.cumsum(t2, 1)
        return t3
# Inputs to the model
x1 = torch.tensor([[0.2332, 1.197, -0.154], [-1.4249, 0.7315, 0.9041]], device='cuda:0')
