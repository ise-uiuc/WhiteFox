
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        b = {}
        a = {}
        b['dtype'] = torch.float32
        b['layout'] = torch.strided
        b['device'] = torch.device('cuda:0')
        a['dtype'] = torch.float64
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:0')
        a['dtype_to'] = torch.float32
        a['dtype_from'] = torch.float64
        b['dtype_to'] = torch.float64
        b['dtype_from'] = torch.float32
        t1 = torch.mul(x1, x2.to(dtype=a['dtype']))
        return t1
# Inputs to the model
x1 = torch.randn(1000, 1000, device='cuda:0')
x2 = torch.randn(1000, 1000, device='cuda:0')
