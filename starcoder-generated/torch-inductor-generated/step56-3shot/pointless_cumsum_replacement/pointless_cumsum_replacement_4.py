
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = torch.int16
        b['layout'] = torch.strided
        b['device'] = torch.device('cuda:0')
        a['dtype'] = torch.uint8
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:0')
        a['dtype_to'] = torch.int16
        a['dtype_from'] = torch.uint8
        b['dtype_to'] = torch.uint8
        b['dtype_from'] = torch.int16
        t1 = torch.full([4, 1761, 164], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = t1.to(dtype=a['dtype'])
        t3 = torch.cumsum(t2, 0)
        t4 = t3.to(dtype=a['dtype_to'])
        return t4
# Inputs to the model
x1 = torch.randn(4, 1761, 164, device='cuda:0')
