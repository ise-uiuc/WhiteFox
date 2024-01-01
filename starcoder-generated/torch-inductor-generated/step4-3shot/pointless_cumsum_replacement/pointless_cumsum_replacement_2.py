
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = torch.bool
        b['layout'] = torch.strided
        b['device'] = torch.device('cuda:0')
        a['dtype'] = torch.long
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:0')
        a['dtype_to'] = torch.int32
        a['dtype_from'] = torch.long
        b['dtype_to'] = torch.bool
        b['dtype_from'] = torch.int32
        t1 = torch.full([2, 4, 8, 256], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = t1.to(dtype=a['dtype'])
        t3 = torch.cumsum(t2, 1)
        t4 = t3.type(torch.float64)
        t5 = t4.type(torch.long)
        t6 = t5.type(torch.float16)
        t7 = torch.zeros_like(t6, dtype=torch.float16)
        return t7
# Inputs to the model
x1 = torch.randn(2, 4, 8, 256, device='cuda:0')
