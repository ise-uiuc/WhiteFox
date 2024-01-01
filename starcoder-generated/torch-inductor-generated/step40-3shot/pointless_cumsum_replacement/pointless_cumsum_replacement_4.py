
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = torch.int16
        b['layout'] = torch.sparse_coo
        b['device'] = torch.device('cuda:0')
        a['dtype'] = torch.bool
        a['layout'] = torch.sparse_coo
        a['device'] = torch.device('cuda:0')
        a['dtype_to'] = torch.int16
        a['dtype_from'] = torch.bool
        b['dtype_to'] = torch.int16
        b['dtype_from'] = torch.bool
        t1 = torch.full([256], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = t1.to(dtype=a['dtype'])
        t3 = torch.cumsum(t2, 0)
        return t3
# Inputs to the model
x1 = torch.randn(256, device='cuda:0', dtype=torch.float)
