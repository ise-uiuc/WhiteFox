
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        a['dtype'] = torch.int8
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:0')
        b['dtype'] = torch.int64
        b['layout'] = torch.strided
        b['device'] = torch.device('cuda:0')
        b['dtype_to'] = torch.int64
        b['dtype_from'] = torch.int8
        t1 = torch.full([64, 64646], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = t1.to(dtype=a['dtype'])
        t3 = torch.cumsum(t2, 1)
        return t3
# Inputs to the model
x1 = torch.randn(64, 64646, device='cuda:0')
