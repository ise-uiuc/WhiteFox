
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
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
        b['dtype_to'] = torch.float32
        b['dtype_from'] = torch.float64
        t1 = torch.full([1, 16], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t1.retain_grad()
        t2 = t1.to(dtype=a['dtype'])
        t3 = torch.tanh(t2)
        return t1, t3
# Inputs to the model
x1 = torch.randn(1, 16, device='cuda:0')
