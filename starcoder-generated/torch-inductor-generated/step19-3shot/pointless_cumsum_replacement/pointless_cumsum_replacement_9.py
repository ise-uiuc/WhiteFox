
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        c = {
          'strides': (1024, 1),
          'requires_grad': False,
            'padding': (0, 0),
            'dilation': (1, 1),
            'is_mkldnn': False,
            'output_padding': (0, 0),
            'groups': 1
        }
        a['dtype'] = torch.float64
        b['dtype'] = torch.float64
        a['shape'] = (4096, 512)
        a['m'] = torch.nn.ConvTranspose2d(a['shape'][1], a['shape'][0], (10, 1), stride=c['strides'], padding=c['padding'], output_padding=c['output_padding'], groups=c['groups'], bias=True, dilation=c['dilation'])
        a['m'].to(a['dtype'])
        b = todevice(b, a['dtype'])
        t1 = torch.full([1024, 512], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = a['m'](t1)
        return b['to'](t2)
# Inputs to the model
x1 = torch.randn(1, 1024, 1, 1, device='cuda:0')
