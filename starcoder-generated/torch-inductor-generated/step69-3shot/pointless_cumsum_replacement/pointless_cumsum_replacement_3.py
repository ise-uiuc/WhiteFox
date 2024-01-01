
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = torch.uint8
        b['layout'] = torch.strided
        b['device'] = torch.device('cuda:0')
        a['dtype'] = torch.int16
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:0')
        a['dtype_to'] = torch.int16
        a['dtype_from'] = torch.int16
        b['dtype_to'] = torch.uint8
        b['dtype_from'] = torch.int16
        t1 = torch.full([512, 64, 64, 2], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = t1.to(dtype=a['dtype'])
        t3 = torch.cumsum(t2, 1)
        return t3
# Inputs to the model
x1 = torch.randn([512, 2, 64, 64], device='cuda:0')
