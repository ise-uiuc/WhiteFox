
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 3, stride=1, padding=0)
    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = torch.uint8
        b['layout'] = torch.strided
        b['device'] = torch.device('cuda:0')
        a['dtype'] = torch.bool
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:0')
        a['dtype_to'] = torch.int32
        a['dtype_from'] = torch.uint8
        b['dtype_to'] = torch.bool
        b['dtype_from'] = torch.int32
        t1 = torch.full([256, 1024], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = t1.to(dtype=a['dtype'])
        t3 = torch.cumsum(t2, 1)
        t4 = t3.type(torch.int32)
        t5 = t4.type(torch.float32)
        t6 = t5.div(1024)
        t7 = t6.type(torch.float32)
        return t7
# Inputs to the model
x1 = torch.randn(256, 1, 1024, 1024, device='cuda:0')
