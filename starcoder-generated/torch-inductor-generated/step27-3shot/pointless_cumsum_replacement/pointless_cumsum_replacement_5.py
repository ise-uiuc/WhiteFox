
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = torch.int32
        b['layout'] = torch.strided
        b['device'] = torch.device('cuda:0')
        a['dtype'] = torch.float16
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:0')
        a['dtype_to'] = torch.float64
        a['dtype_from'] = torch.float16
        b['dtype_to'] = torch.float32
        b['dtype_from'] = torch.float64
        t1 = torch.full([64, 8], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = convert_element_type(t1, b['dtype_to'])
        t3 = torch.cumsum(t2, 0)
        t4 = convert_element_type(t3, b['dtype_from'])
        return t4
# Inputs to the model
x1 = torch.randn(64, 8, device='cuda:0')
