
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = torch.long
        b['layout'] = torch.sparse_coo
        b['device'] = torch.device('cuda:0')
        a['dtype'] = torch.int8
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:0')
        a['dtype_to'] = torch.long
        a['dtype_from'] = torch.int8
        b['dtype_to'] = torch.long
        b['dtype_from'] = torch.int8
        t1 = torch.full([41, 32768], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = t1.to(dtype=a['dtype'])
        t3 = torch.cumsum(t2, 1)
        return t3
# Inputs to the model
x1 = torch.randn(41, 32768, device='cuda:0', dtype=torch.int64)
