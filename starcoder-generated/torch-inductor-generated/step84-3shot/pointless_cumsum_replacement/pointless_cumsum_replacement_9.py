
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = torch.bool
        b['layout'] = torch.strided
        b['device'] = torch.device('cpu')
        a['dtype'] = torch.float16
        a['layout'] = torch.strided
        a['device'] = torch.device('cpu')
        a['dtype_to'] = torch.uint8
        a['dtype_from'] = torch.float16
        b['dtype_to'] = torch.uint8
        b['dtype_from'] = torch.float16
        t1 = torch.rand([512, 1, 128], dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = t1.sum(2).to(dtype=a['dtype'])
        t3 = torch.cumsum(t2, 1)
        return t3
# Inputs to the model
x1 = torch.randn(512, 1, 128, device='cpu')
