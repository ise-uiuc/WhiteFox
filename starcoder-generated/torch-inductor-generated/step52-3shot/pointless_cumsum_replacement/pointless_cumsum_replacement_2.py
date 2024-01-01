
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        a['torch.return_types.conv2d'] = torch.return_types.conv2d
        b['stride'] = [1, 1]
        a['stride'] = [1, 1]
        b['dilation'] = [1, 1]
        a['dilation'] = [1, 1]
        b['dtype'] = torch.float32
        b['layout'] = torch.strided
        b['device'] = torch.device('cuda:0')
        a['dtype'] = torch.float32
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:0')
        a['weight_dtype'] = torch.float32
        a['bias_dtype'] = torch.float32
        b['weight_dtype'] = torch.float32
        b['bias_dtype'] = torch.float32
        a['groups'] = 1
        b['groups'] = 1
        t1 = torch.full([16, 3, 16, 32], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = t1.to(dtype=a['dtype'])
        t3 = torch.flip(t1, [1])
        t4 = torch.flip(t2, [1])
        t5 = torch.cumsum(t3, 2)
        t6 = torch.cumsum(t4, 2)
        t7 = torch.conv2d(t5, t6, None, stride=a['stride'], padding=[1, 1], dilation=a['dilation'], groups=a['groups'])
        t8 = torch.cumsum(t6, 2)
        t9 = torch.flip(t7, [2])
        t10 = torch.cumsum(t8, 1)
        t11 = torch.flip(t9, [2])
        t12 = torch.conv2d(t10, t10, None, stride=[1, 1], padding=[1, 1], dilation=b['dilation'], groups=b['groups'])
        t13 = torch.cumsum(t10, 1)
        t14 = torch.conv2d(t12, t10, None, stride=[1, 1], padding=[1, 1], dilation=b['dilation'], groups=b['groups'])
        t15 = torch.cumsum(t11, 1)
        t16 = torch.conv2d(t13, t11, None, stride=[1, 1], padding=[1, 1], dilation=b['dilation'], groups=b['groups'])
        t17 = torch.cumsum(t14, 1)
        t18 = torch.conv2d(t16, t15, None, stride=[1, 1], padding=[1, 1], dilation=b['dilation'], groups=b['groups'])
        return t18
# Inputs to the model
x1 = torch.randn(16, 3, 16, 32, device='cuda:0')
