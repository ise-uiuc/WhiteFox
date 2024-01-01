
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = torch.float16
        b['layout'] = torch.strided
        b['device'] = torch.device('cuda:0')
        a['dtype'] = torch.bfloat16
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:0')
        a['dtype_to'] = torch.float16
        a['dtype_from'] = torch.bfloat16
        b['dtype_to'] = torch.float16
        b['dtype_from'] = torch.bfloat16
        t1 = torch.full([512, 1200], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = t1.to(dtype=a['dtype'])
        t3 = torch.cumsum(t2, 0)
        t4 = t3.to(device='cuda:0')
        return t4
# Inputs to the model
x1 = torch.randn(512, 1200, device='cuda:0')
