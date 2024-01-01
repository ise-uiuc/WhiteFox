
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
        a['dtype_to'] = torch.int64
        a['dtype_from'] = torch.int16
        b['dtype_to'] = torch.float16
        b['dtype_from'] = torch.int32
        t1 = torch.full([128, 512], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = t1.to(dtype=a['dtype'])
        t3 = torch.abs(t2)
        t4 = torch.mean(t3, -1)
        t5 = torch.neg(t4)
        return t5
# Inputs to the model
x1 = torch.randn(128, 512, dtype = torch.float16, device='cuda:0')
