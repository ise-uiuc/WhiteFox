
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        b['dtype'] = torch.float32
        b['layout'] = torch.strided
        b['device'] = torch.device('cpu')
        b['dtype_to'] = torch.float32
        b['dtype_from'] = torch.int64
        a = {}
        a['dtype'] = torch.int64
        a['layout'] = torch.strided
        a['device'] = torch.device('cpu')
        a['dtype_to'] = torch.float32
        a['dtype_from'] = torch.int64
        t1 = torch.full([1024], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t3 = t1.to(dtype=a['dtype'])
        return t3
# Inputs to the model
x1 = torch.randn(1024, device='cpu')
