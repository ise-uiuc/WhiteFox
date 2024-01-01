
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b1 = {}
        a = {}
        b1['dtype'] = torch.float32
        b1['layout'] = torch.strided
        b1['device'] = torch.device('cpu')
        a['dtype'] = torch.float32
        a['layout'] = torch.strided
        a['device'] = torch.device('cpu')
        a['dtype_to'] = torch.float32
        a['dtype_from'] = torch.float32
        b1['dtype_to'] = torch.float32
        b1['dtype_from'] = torch.float32
        t1 = torch.full([256, 16], 1, dtype=b1['dtype'], layout=b1['layout'], device=b1['device'], pin_memory=False)
        t2 = torch.cat((torch.transpose(t1, 0, 1), torch.transpose(t1, 0, 1)), 0)
        t3 = t2.to(dtype=a['dtype'])
        t4 = torch.cumsum(t3, 0)
        return t4
# Inputs to the model
x1 = torch.randn(256, 16, device='cpu')
