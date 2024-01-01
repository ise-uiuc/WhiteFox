
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = torch.zeros(128).dtype
        b['layout'] = torch.strided
        b['device'] = torch.device('cuda:0')
        a['dtype'] = torch.zeros(128).dtype
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:0')
        a['dtype_to'] = torch.zeros(128).dtype
        a['dtype_from'] = torch.zeros(128).dtype
        b['dtype_to'] = torch.zeros(128).dtype
        b['dtype_from'] = torch.zeros(128).dtype
        t1 = torch.full([256, 1024], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = t1.to(dtype=a['dtype'])
        t3 = torch.cumsum(t2, 1)
        return t3
# Inputs to the model
x1 = torch.randn(256, 1024, device='cuda:0')
