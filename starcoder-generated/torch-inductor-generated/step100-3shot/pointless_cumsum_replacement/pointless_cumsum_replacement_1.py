
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        a = {}
        b = {}
        a['dtype'] = torch.float16
        a['layout'] = torch.strided
        a['device'] = torch.device('cpu')
        a['dtype_to'] = torch.float16
        b['dtype'] = torch.int32
        b['layout'] = torch.strided
        b['device'] = torch.device('cpu')
        b['dtype_to'] = torch.float16
        t1 = torch.full([256, 2048], 1, dtype=a['dtype'], layout=a['layout'], device=a['device'], pin_memory=False)
        t2 = t1.to(dtype=b['dtype'])
        t3 = torch.cumsum(t2, 1)
        return t3
# Inputs to the model
x1 = torch.randn(256, 2048, device='cpu')
