
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        c = {}
        b['dtype'] = torch.uint8
        b['layout'] = torch.strided
        b['device'] = torch.device('cuda:0')
        b['dtype_to'] = torch.bool
        c['dtype'] = torch.uint8
        c['layout'] = torch.strided
        c['device'] = torch.device('cuda:0')
        c['dtype_to'] = torch.bool
        t1 = torch.full([64, 256], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = t1.to(dtype=torch.float32)
        t3 = t2.floor()
        t4 = t3.to(dtype=c['dtype'])
        t5 = t4.to(dtype=torch.bool)
        return t5
# Inputs to the model
x1 = torch.randn(64, 256, device='cuda:0')
