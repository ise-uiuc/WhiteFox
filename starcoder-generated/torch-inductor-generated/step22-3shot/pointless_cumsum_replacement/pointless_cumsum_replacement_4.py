
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = torch.float32
        b['layout'] = torch.strided
        b['device'] = torch.device('cpu')
        a['dtype'] = torch.uint8
        a['layout'] = torch.strided
        a['device'] = torch.device('cpu')
        a['dtype_to'] = torch.float32
        a['dtype_from'] = torch.uint8
        b['dtype_to'] = torch.float32
        b['dtype_from'] = torch.uint8
        t1 = torch.full([1000, 1000], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = t1.to(memory_format=torch.channels_last)
        t4 = torch.view_as_real(t2)
        t3 = torch.cumsum(t4, 2)
        return t3
# Inputs to the model
x1 = torch.randn(5, 2, 3, 3, device='cpu')
