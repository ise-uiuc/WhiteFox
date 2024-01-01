
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = torch.uint8
        b['layout'] = torch.strided
        b['device'] = torch.device('cuda:0')
        a['dtype'] = torch.float16
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:0')
        a['dtype_to'] = torch.float16
        a['dtype_from'] = torch.uint8
        b['dtype_to'] = torch.float16
        b['dtype_from'] = torch.float16
        t1 = torch.full([1, 1024], 1.0, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = t1.to(dtype=a['dtype'])
        t3 = torch.cumsum(t2, 1)
        return t3
# Inputs to the model
x1 = torch.tensor([1.1961, -1.247, -0.3094, -1.0108, 0.5122, 0.7516, 0.2289, 0.8819, 1.4993, 1.4185], device='cuda:0')
