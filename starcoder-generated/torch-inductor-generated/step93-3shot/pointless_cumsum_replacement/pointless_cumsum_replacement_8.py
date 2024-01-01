
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = torch.float32
        b['layout'] = torch.strided
        b['device'] = torch.device('cuda:0')
        a['dtype'] = torch.uint
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:0')
        a['dtype_to'] = torch.float32
        a['dtype_from'] = torch.int64
        b['dtype_to'] = torch.float64
        b['dtype_from'] = torch.double
        t1 = torch.full([33, 4], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = convert_element_type(t1, a['dtype'])
        t3 = t2.to(dtype=a['dtype_to'])
        t4 = torch.cumsum(t3, 1)
        return t4
# Inputs to the model
x1 = torch.randn(33, 4, device='cuda:0')
