
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        a = {}
        a['dtype'] = torch.int8
        a['layout'] = torch.strided
        a['device'] = torch.device('cpu')
        a['dtype_to'] = torch.int16
        a['dtype_from'] = torch.int8
        t1 = torch.full([128, 1024], 1, dtype=a['dtype'], layout=a['layout'], device=a['device'], pin_memory=False)
        t2 = t1.to(dtype=a['dtype_to'])
        t3 = torch.bitwise_xor(t2, t1)
        return torch.isfinite(t3)


# Inputs to the model
x1 = torch.randn(128, 1024, device='cpu')
