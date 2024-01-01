
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Conv2d(3, 3, 2, stride=1, padding=0)
    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = torch.int8
        b['layout'] = torch.strided
        b['device'] = torch.device('cuda:0')
        a['dtype'] = torch.float32
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:0')
        a['dtype_to'] = torch.float16
        a['dtype_from'] = torch.float32
        b['dtype_to'] = torch.float32
        b['dtype_from'] = torch.float16
        t1 = torch.full([512, 512], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = t1.to(dtype=a['dtype'])
        t3 = torch.cumsum(t2, 0)
        t4 = t3.type(torch.float16)
        t5 = t4.type(torch.int16)
        t6 = t5.div(1024)
        t7 = t6.type(torch.float32)
        return t7
# Inputs to the model
x1 = torch.randn(512, 512, device='cuda:0')
