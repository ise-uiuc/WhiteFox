
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(1, 1, kernel_size=(1,), stride=(1,))
    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = torch.bool
        b['layout'] = torch.strided
        b['device'] = torch.device('cuda:0')
        a['dtype'] = torch.uint8
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:0')
        a['dtype_to'] = torch.byte
        a['dtype_from'] = torch.uint8
        b['dtype_to'] = torch.uint8
        b['dtype_from'] = torch.bool
        t1 = torch.full([1, 1], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = t1.to(dtype=a['dtype'])
        t3 = torch.cumsum(t2, 1)
        t4 = self.conv2d(t2)
        return t4
# Inputs to the model
x1 = torch.randn(1, 1, device='cuda:0')
