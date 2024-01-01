
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
        a['device'] = torch.device('cpu')
        b['dtype_to'] = torch.float64
        b['dtype_from'] = torch.float64
        t1 = torch.full([2048, 512], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=True)
        t2 = t1.to(dtype=a['dtype'])
        t3 = torch.cumsum(t2, 1)
        return t3
# Inputs to the model
x1 = torch.randn(2048, 512, device='cuda:0')
