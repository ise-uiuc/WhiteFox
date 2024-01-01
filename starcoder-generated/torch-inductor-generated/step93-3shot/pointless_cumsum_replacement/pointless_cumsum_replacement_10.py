
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(2, 3, 2)
    def forward(self, x1):
        a = {}
        a['dtype'] = torch.uint8
        a['layout'] = torch.strided
        a['device'] = torch.device('cpu')
        a['dtype_to'] = torch.int64
        a['dtype_from'] = torch.int64
        t1 = torch.full([6, 16, 16], 1, dtype=a['dtype'], layout=a['layout'], device=a['device'], pin_memory=False)
        t2 = t1.to(dtype=torch.int64)
        t3 = self.conv(t2)
        return t3
# Inputs to the model
x1 = torch.randn(6, 16, 2, device='cpu')
