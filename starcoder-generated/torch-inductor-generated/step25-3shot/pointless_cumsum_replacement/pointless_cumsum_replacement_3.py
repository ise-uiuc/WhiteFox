
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        b = {}
        a = {}
        b['dtype'] = torch.int32
        b['layout'] = torch.strided
        b['device'] = torch.device('cpu')
        a['dtype'] = torch.float64
        a['layout'] = torch.strided
        a['device'] = torch.device('cpu')
        c = torch.addmm(x1, x2, x2, beta=0, alpha=1)
        d = torch.sum(c, dtype=b['dtype'], layout=b['layout'], device=b['device'])
        return d
# Inputs to the model
x1 = torch.randn(2048, 9, device='cpu')
x2 = torch.randn(2048, 1024, device='cpu')
