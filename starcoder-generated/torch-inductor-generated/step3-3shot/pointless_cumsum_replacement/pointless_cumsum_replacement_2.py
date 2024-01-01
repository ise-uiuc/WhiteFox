
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = torch.float16
        b['layout'] = torch.strided
        b['device'] = torch.device('cuda:0')
        a['dtype'] = torch.float16
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:0')
        a['dtype_to'] = torch.bool
        a['dtype_from'] = torch.float16
        b['dtype_to'] = torch.bool
        b['dtype_from'] = torch.float16
        t1 = torch.full([512, 32], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = torch.cumsum(t1, 1)
        t3 = t2.to(dtype=a['dtype'])
        t4 = torch.cumsum(t3, 1)
        return t4
# Inputs to the model
x1 = torch.randn(512, 32, device='cuda:0')
