
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        b = {}
        a = {}
        b['dtype'] = torch.uint8
        b['layout'] = torch.strided
        b['device'] = torch.device('cpu')
        a['dtype'] = torch.bool
        a['layout'] = torch.strided
        a['device'] = torch.device('cpu')
        a['dtype_to'] = torch.bool
        a['dtype_from'] = torch.uint8
        b['dtype_to'] = torch.int32
        b['dtype_from'] = torch.uint8
        t1 = torch.full([1, 178], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = t1.to(dtype=a['dtype'])
        t3 = torch.narrow(t2, 1, 0, x2)
        t4 = torch.cumsum(t3, 1)
        t5 = t4.to(dtype=a['dtype_to'])
        t6 = t5.to(dtype=a['dtype_to'])
        return t6
# Inputs to the model
x1 = torch.randn(1, 178, device='cpu')
x2 = torch.abs(torch.randn(1, dtype=torch.int32)) + 5
