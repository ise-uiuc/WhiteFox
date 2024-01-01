
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        a = {}
        a['dtype'] = torch.int16
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:0')
        a['dtype_to'] = torch.int16
        a['dtype_from'] = torch.int32
        t1 = x1.to(dtype=a['dtype'])
        t2 = t1.to(dtype=a['dtype_to'])
        t3 = torch.cumsum(t2, 1)
        return t3
# Inputs to the model
x1 = torch.randn(1, 2048, device='cuda:0')
