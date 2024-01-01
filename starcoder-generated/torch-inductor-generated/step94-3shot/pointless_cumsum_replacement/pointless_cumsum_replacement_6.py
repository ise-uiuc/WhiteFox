
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = torch.float32
        b['shape'] = torch.Size([2, 1, 8, 8])
        b['layout'] = torch.strided
        b['device'] = torch.device('cuda:0')
        a['dtype'] = torch.float32
        a['shape'] = torch.Size([2, 8, 8])
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:0')
        a['dtype_to'] = torch.float32
        a['dtype_from'] = torch.float32
        b['dtype_to'] = torch.bfloat16
        b['dtype_from'] = torch.uint8
        t1 = torch.full(b['shape'], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = t1.to(dtype=a['dtype'])
        t3 = torch.cumsum(t2, 1)
        return t3
# Inputs to the model
x1 = torch.full([2, 1, 8, 8], 1, dtype=torch.uint8, device='cuda:0')
