
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        a = {}
        a['dtype'] = torch.float32
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:0')
        a['dtype_to'] = torch.int16
        a['dtype_from'] = torch.float32
        t1 = torch.full([1024, 128], 1, dtype=a['dtype'], layout=a['layout'], device=a['device'], pin_memory=False)
        t2 = t1.to(dtype=a['dtype_to'])
        t3 = torch.cumsum(t2, 1)
        return t3
# Inputs to the model
x1 = torch.randn(1024, 128, device='cuda:0')
