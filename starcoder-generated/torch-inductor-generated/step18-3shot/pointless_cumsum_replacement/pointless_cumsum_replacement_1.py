
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        a['dtype'] = torch.float16
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:0')
        b['dtype'] = torch.bool
        b['layout'] = torch.strided
        b['device'] = torch.device('cuda:0')
        a['dtype_to'] = torch.float16
        a['dtype_from'] = torch.float16
        b['dtype_to'] = torch.float16
        b['dtype_from'] = torch.float16
        t1 = torch.full([1, 1024], -0.00085, dtype=a['dtype'], layout=a['layout'], device=a['device'], pin_memory=False)
        t2 = t1.to(dtype=b['dtype'])
        t3 = torch.cumsum(t2, 0)
        return t3
# Inputs to the model
x1 = torch.tensor(-0.00085, device='cuda:0')
