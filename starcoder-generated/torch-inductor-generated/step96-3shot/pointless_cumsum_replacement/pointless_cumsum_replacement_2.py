
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = torch.device('cuda:0')
        b['layout'] = torch.strided
        b['device'] = torch.device('cpu')
        a['dtype'] = torch.device('cuda:0')
        a['layout'] = torch.strided
        a['device'] = torch.device('cpu')
        a['dtype_to'] = torch.device('cuda:0')
        a['dtype_from'] = torch.device('cuda:0')
        b['dtype_to'] = torch.device('cpu')
        b['dtype_from'] = torch.device('cpu')
        t1 = torch.full([1, 1024], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = t1.to(dtype=a['dtype'])
        t3 = torch.cumsum(t2, 1)
        return t3
# Inputs to the model
x1 = torch.randn(1, 1024, device='cuda:0')
