
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = torch.int16
        b['layout'] = torch.complex_float32
        b['device'] = torch.device('cpu')
        a['dtype'] = torch.float32
        a['layout'] = torch.strided
        a['device'] = torch.device('cpu')
        a['dtype_from'] = torch.int16
        b['dtype_to'] = torch.float32
        b['dtype_from'] = torch.float32
        t1 = torch.full([512, 512], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = t1 + 1
        t3 = torch.cumsum(t2, 1)
        return t3
# Inputs to the model
x1 = torch.randn(512, 512, device='cuda:0')
