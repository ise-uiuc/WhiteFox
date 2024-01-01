
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 3, stride=1, padding=0)
    def forward(self, x1):
        b1 = {}
        a1 = {}
        b2 = {}
        a2 = {}
        b1['dtype'] = torch.bool
        b1['layout'] = torch.strided
        b1['device'] = torch.device('cuda:0')
        a1['dtype'] = torch.float32
        a1['layout'] = torch.strided
        a1['device'] = torch.device('cuda:0')
        a1['dtype_to'] = torch.float16
        a1['dtype_from'] = torch.float32
        b1['dtype_to'] = torch.float32
        b1['dtype_from'] = torch.float16
        b2['dtype'] = torch.float32
        b2['layout'] = torch.strided
        b2['device'] = torch.device('cuda:0')
        a2['dtype'] = torch.float16
        a2['layout'] = torch.strided
        a2['device'] = torch.device('cuda:0')
        a2['dtype_to'] = torch.float32
        a2['dtype_from'] = torch.float16
        t1 = torch.full([4096, 512], 1, dtype=b1['dtype'], layout=b1['layout'], device=b1['device'], pin_memory=False)
        t2 = t1.to(dtype=a1['dtype'])
        t3 = torch.cumsum(t2, 1)
        t4 = t3.type(torch.float16)
        t5 = t4.type(torch.int16)
        t6 = t5.div(512)
        t7 = t6.type(torch.float32)
        t8 = t7.type(torch.uint16)
        t9 = t8.to(dtype=a2['dtype'])
        t10 = torch.sum(t9)
        return t10
# Inputs to the model
x1 = torch.randn(4096, 1, 256, 256, device='cuda:0')
