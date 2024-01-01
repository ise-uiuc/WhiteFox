
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b1 = {}
        b2 = {}
        b3 = {}
        b4 = {}
        b1['dtype'] = torch.bool
        b1['layout'] = torch.strided
        b2['dtype'] = torch.bool
        b2['layout'] = torch.strided
        b3['dtype'] = torch.bool
        b3['layout'] = torch.strided
        b4['dtype'] = torch.bool
        b4['layout'] = torch.strided
        t1 = x1.to(dtype=b1['dtype'])
        t2 = x1.to(dtype=b2['dtype'])
        t3 = torch.sqrt(t1)
        t4 = x1.to(dtype=b4['dtype'])
        t5 = torch.sqrt(t4)
        t6 = x1.to(dtype=b3['dtype'])
        t7 = torch.sqrt(t6)
        t8 = torch.mul(t5, t7)
        return torch.mul(t2, t8)
# Inputs to the model
x1 = torch.randn(1024, 1024)
