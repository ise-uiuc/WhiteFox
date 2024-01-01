
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = torch.float64
        b['layout'] = torch.sparse_coo
        b['device'] = torch.device('cuda:1')
        a['dtype'] = torch.float32
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:1')
        a['dtype_to'] = torch.int64
        a['dtype_from'] = torch.float32
        b['dtype_to'] = torch.int64
        b['dtype_from'] = torch.float64
        t0 = x1.to(dtype=torch.int64)
        t1 = torch.full([64, 2048], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = t1.to(dtype=a['dtype'])
        t3 = torch.cumsum(t2, 1)
        return t3
# Inputs to the model
x1 = torch.transpose(torch.randint(1, 2, [128, 2048], dtype=torch.int64), 1, 0)
