
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = torch.float16
        b['layout'] = torch.strided
        b['device'] = torch.device('cuda:0')
        a['dtype'] = torch.float16
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:0')
        a['dtype_to'] = torch.float32
        a['dtype_from'] = torch.float16
        b['dtype_to'] = torch.float32
        b['dtype_from'] = torch.float16
        t1 = torch.full([2048, 128], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = t1.to(dtype=a['dtype'])
        t3 = torch.empty_like(t2)
        t4 = torch.cumsum(t2, 1, out=t3)
        return t4
# Inputs to the model
x1 = torch.randn(2048, 128, dtype=torch.float16, device='cuda:0')
