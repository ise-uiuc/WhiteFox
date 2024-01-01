
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        b = {}
        a = {}
        b['dtype'] = torch.int32
        b['layout'] = torch.strided
        b['device'] = torch.device('cpu')
        a['dtype'] = torch.float32
        a['layout'] = torch.strided
        a['device'] = torch.device('cpu')
        a['dtype_to'] = torch.int32
        a['dtype_from'] = torch.float32
        b['dtype_to'] = torch.int32
        b['dtype_from'] = torch.int32
        t1 = torch.full([256, 256], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = x1.to(dtype=a['dtype'])
        t3 = t2 + t1
        return t3
# Inputs to the model
x1 = torch.randn(256, 256, device='cpu')
x2 = 1
