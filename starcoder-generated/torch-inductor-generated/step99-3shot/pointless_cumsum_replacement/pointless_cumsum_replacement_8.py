
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = torch.float16
        b['layout'] = torch.strided
        b['device'] = torch.device('cpu')
        a['dtype'] = torch.uint8
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:0')
        a['dtype_to'] = torch.uint8
        a['dtype_from'] = torch.float16
        b['dtype_to'] = torch.uint8
        b['dtype_from'] = torch.float16
        t1 = torch.full([1, 1], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        y = x1.to(device=a['device'])
        t2 = torch.add(t1, y)
        t3 = t2.to(dtype=a['dtype'])
        t4 = t3.to(device=b['device'])
        t5 = t4.permute(0, 1)
        t6 = torch.cumsum(t5, 1)
        return t6
# Inputs to the model
x1 = torch.randn(1, 1, device='cuda:0')
