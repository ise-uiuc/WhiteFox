
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        a = {}
        b = {}
        a['dtype'] = torch.float
        b['dtype'] = torch.float
        a['layout'] = torch.strided
        b['layout'] = torch.strided
        a['device'] = torch.device('cuda:0')
        b['device'] = torch.device('cuda:0')
        a['dtype_to'] = torch.float
        b['dtype_to'] = torch.float
        a['dtype_from'] = torch.long
        b['dtype_from'] = torch.long
        t1 = torch.full([64, 64], 1, dtype=a['dtype'], layout=a['layout'], device=a['device'], pin_memory=False)
        t2 = t1.to(dtype=a['dtype_to'])
        t3 = t2.clamp_(min=0, max=2)
        t4 = torch.ceil(t3)
        t5 = torch.as_strided(t4, size=[64, 64], stride=[0, 0])
        t6 = t5.to(dtype=b['dtype_from'])
        return t6
# Inputs to the model
x1 = torch.randn(64, 64, device='cuda:0')
