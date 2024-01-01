
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = torch.float16
        b['layout'] = torch.sparse_coo
        b['device'] = torch.device('cuda:2')
        a['dtype'] = torch.int32
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:2')
        a['dtype_to'] = torch.float16
        a['dtype_from'] = torch.int32
        b['dtype_to'] = torch.float16
        b['dtype_from'] = torch.int32
        t1 = torch.full([33, 256], 10000, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = t1.to(dtype=a['dtype'])
        t3 = torch.cumsum(t2, 1)
        return t3
# Inputs to the model
x1 = torch.randint(0, 10000, [33, 256], dtype=torch.int32, device='cuda:0')
