
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, (2, 2), (1, 1), (0, 0), 1, 1, False, [1, 1], 1)
    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = torch.float64
        b['layout'] = torch.strided
        b['device'] = torch.device('cpu')
        a['dtype'] = torch.int64
        a['layout'] = torch.strided
        a['device'] = torch.device('cpu')
        a['dtype_to'] = torch.int64
        a['dtype_from'] = torch.float64
        b['dtype_to'] = torch.int64
        b['dtype_from'] = torch.float64
        t1 = torch.full([1080, 1920], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = t1.to(dtype=a['dtype'])
        t3 = self.conv(t2)
        t4 = torch.cumsum(t3, 1)
        t5 = torch.cumsum(t4, 2)
        return t5
# Inputs to the model
x1 = torch.randn(1080, 1920, device='cpu')
