
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = torch.float64
        b['layout'] = torch.strided
        b['device'] = torch.device('cuda:0')
        a['dtype'] = torch.float64
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:0')
        a['dtype_to'] = torch.float64
        a['dtype_from'] = torch.float64
        b['dtype_to'] = torch.float64
        b['dtype_from'] = torch.float64
        t1 = torch.full([1, 512], 1, dtype=a['dtype'], layout=a['layout'], device=a['device'], pin_memory=False)
        t2 = t1
        t3 = torch.cumsum(t2, 0)
        return t3
# Inputs to the model
x1 = torch.randn(1, 512, device='cuda:0')
