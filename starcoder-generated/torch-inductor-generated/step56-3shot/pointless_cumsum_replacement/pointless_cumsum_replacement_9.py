
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = torch.float16
        b['layout'] = torch.strided
        a['dtype'] = torch.float16
        a['layout'] = torch.strided
        b['device'] = torch.device('cuda:0')
        a['device'] = torch.device('cuda:0')
        a['dim1'] = 0
        a['dim2'] = 1
        a['dtype_to'] = torch.float16
        a['dtype_from'] = torch.float16
        b['dim_to'] = 0
        b['dim_from'] = 1
        b['dtype_to'] = torch.float16
        b['dtype_from'] = torch.float16
        t1 = torch.arange(9, dtype=b['dtype'], layout=b['layout'], device=b['device'])
        t2 = torch.transpose(t1, b['dim_from'], b['dim_to'])
        t3 = torch.sum(t2)
        return t3.to(dtype=a['dtype'])
# Inputs to the model
x1 = torch.randn(20, 30, device='cuda:0')
