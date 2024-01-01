
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 3, stride=1, padding=0)
    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = torch.float32
        b['layout'] = torch.strided
        b['device'] = torch.device('cuda:0')
        a['dtype'] = torch.float32
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:0')
        a['dtype_to'] = torch.float16
        a['dtype_from'] = torch.float32
        b['dtype_to'] = torch.float32
        b['dtype_from'] = torch.float16
        t1 = torch.full([256, 3, 2560, 2048], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = t1.to(dtype=a['dtype'])
        t3 = t2.sum(dim=2)
        t4 = torch.cumsum(t3, 1)
        t5 = t4.type(torch.float16)
        t6 = t5.div(2560)
        return t6
# Inputs to the model
x1 = torch.randn(256, 1, 2560, 2048, device='cuda:0')
