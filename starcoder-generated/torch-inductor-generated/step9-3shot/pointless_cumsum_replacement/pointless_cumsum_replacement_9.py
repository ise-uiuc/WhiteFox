
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = torch.uint8
        b['layout'] = torch.strided
        b['device'] = torch.device('cuda:0')
        a['dtype'] = torch.double
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:0')
        a['dtype_to'] = torch.uint8
        a['dtype_from'] = torch.double
        b['dtype_to'] = torch.double
        b['dtype_from'] = torch.uint8
        t1 = torch.full([512, 1024], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = torch.cumsum(t1, 1)
        t3 = t2.to(dtype=a['dtype'])
        return t3
# Inputs to the model
x1 = torch.randn(512, 1024, device='cuda:0')
