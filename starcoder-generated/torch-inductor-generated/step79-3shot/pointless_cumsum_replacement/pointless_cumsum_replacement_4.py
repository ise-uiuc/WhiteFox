
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = torch.float32
        b['layout'] = torch.strided
        b['device'] = torch.device('cpu')
        a['dtype'] = torch.float16
        a['layout'] = torch.strided
        a['device'] = torch.device('cpu')
        a['dtype_to'] = torch.float32
        a['dtype_from'] = torch.float16
        a = t1.to(dtype=a['type'], device=a['device'], layout=a['layout'])
        t2 = t1.to(dtype=a['dtype_to'])
        t3 = torch.cumsum(t2, 1)
        return t3
# Inputs to the model
x1 = torch.randn(256, 256, device='cpu')
