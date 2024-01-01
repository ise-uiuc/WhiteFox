
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = torch.bool
        b['layout'] = torch.strided
        b['device'] = torch.device('cuda:0')
        a['dtype'] = torch.float32
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:0')
        a['dtype_to'] = torch.float32
        b['dtype_to'] = torch.float64
        b['dtype_from'] = torch.float32
        t1 = torch.full([8, 1, 2], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = t1.view(1,8,2)
        t3 = t2.to(dtype=a['dtype'])
        t4 = torch.cumsum(t3, -1)
        return t4
# Inputs to the model
x1 = torch.randn(1, 8, 2, device='cuda:0')
