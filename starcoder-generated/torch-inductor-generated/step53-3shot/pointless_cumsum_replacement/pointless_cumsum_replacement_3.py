
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = torch.float32
        b['layout'] = torch.strided
        b['device'] = torch.device('cuda:0')
        a['dtype'] = torch.int64
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:0')
        a['dtype_to'] = torch.int64
        a['dtype_from'] = torch.float32
        b['dtype_to'] = torch.float32
        b['dtype_from'] = torch.int64
        t1 = torch.full([8192, 500], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = torch.full([8192, 500], 2, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t3 = torch.max(t1, t2)
        t4 = t3.to(dtype=a['dtype'])
        t5 = torch.cumsum(t4, 1)
        return t5
# Inputs to the model
x1 = torch.randn(8192, 500, device='cuda:0')
