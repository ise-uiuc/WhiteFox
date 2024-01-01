
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        a['dtype'] = torch.uint8
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:0')
        b['dtype'] = torch.float32
        b['layout'] = torch.strided
        b['device'] = torch.device('cuda:0')
        a['dtype_to'] = torch.int32
        a['dtype_from'] = torch.int32
        b['dtype_to'] = torch.float64
        b['dtype_from'] = torch.float32
        t1 = torch.full([256, 1024], 0, dtype=a['dtype'], layout=a['layout'], device=a['device'], pin_memory=False)
        t2 = t1.to(b['dtype'])
        t3 = torch.cumsum(t2, 1)
        return t3
# Inputs to the model
x1 = torch.ones(256, 1024, dtype=torch.uint8, device='cuda:0')
