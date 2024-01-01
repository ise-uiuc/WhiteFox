
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        b = {}
        a = {}
        b['dtype'] = torch.float32
        b['layout'] = torch.strided
        b['device'] = torch.device('cuda:0')
        a['dtype'] = torch.float32
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:0')
        a['dtype_to'] = torch.float32
        a['dtype_from'] = torch.float32
        b['dtype_to'] = torch.int64
        b['dtype_from'] = torch.float32
        t0 = torch.full([128, 1156759], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t1 = torch.rand([256, 256], dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = torch.full([256, 256], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t3 = torch.matmul(t0, t1)
        return t3
# Inputs to the model
x1 = 256
x2 = 128
