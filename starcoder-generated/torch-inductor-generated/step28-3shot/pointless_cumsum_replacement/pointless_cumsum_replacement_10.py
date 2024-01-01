
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = torch.int8
        b['layout'] = torch.strided
        b['device'] = torch.device('cuda:0')
        a['dtype'] = torch.uint8
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:0')
        a['dtype_to'] = torch.uint8
        a['dtype_from'] = torch.int8
        t1 = torch.full([256, 512], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = torch.quantize_per_tensor(t1, 0.0, 255, torch.quint8)
        t3 = t2.dequantize()
        t4 = t3.to(dtype=a['dtype'])
        return t4
# Inputs to the model
x1 = torch.randn(256, 512, device='cuda:0')
