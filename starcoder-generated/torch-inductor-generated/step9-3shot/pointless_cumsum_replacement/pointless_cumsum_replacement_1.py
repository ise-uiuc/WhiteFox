
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = torch.float32
        b['layout'] = torch.strided
        b['device'] = torch.device('cuda:1')
        a['dtype'] = torch.half
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:0')
        a['dtype_to'] = torch.half
        a['dtype_from'] = torch.float32
        b['dtype_to'] = torch.half
        b['dtype_from'] = torch.float32
        t1 = torch.full([256, 1024], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = t1.to(dtype=a['dtype'])
        t3 = torch.cumsum(t2, 1)
        return t3
# Inputs to the model
x1 = torch.randn(256, 1024, device='cuda:1')
