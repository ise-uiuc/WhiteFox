
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        b = {}
        a = {}
        b['dtype'] = torch.int64
        b['layout'] = torch.strided
        b['device'] = torch.device('cuda:0')
        a['dtype'] = torch.float64
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:0')
        a['dtype_to'] = torch.float64
        a['dtype_from'] = torch.int64
        b['dtype_to'] = torch.int64
        b['dtype_from'] = torch.float64
        t1 = x.to(dtype=a['dtype'])
        t2 = torch.cumsum(t1, 0)
        return t2
# Inputs to the model
x = torch.randint(10, (512, 1024), dtype=b['dtype'], layout=b['layout'], device=b['device'])
