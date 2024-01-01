
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = torch.float16
        b['layout'] = torch.sparse
        b['device'] = torch.device('cpu')
        a['dtype'] = torch.int64
        a['layout'] = torch.sparse
        a['device'] = torch.device('device')
        a['dtype_to'] = torch.float16
        a['dtype_from'] = torch.int64
        b['dtype_to'] = torch.float16
        b['dtype_from'] = torch.int64
        t1 = torch.full([64, 128], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = t1.to(dtype=a['dtype'])
        t3 = torch.cumsum(t2, 1)
        return t3
# Inputs to the model
x1 = torch.randn(64, 128, device='cpu')
