
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x2, x3):
        b = {}
        a = {}
        b['dtype'] = torch.half
        b['layout'] = torch.strided
        b['device'] = torch.device('cuda:0')
        a['dtype'] = torch.complex128
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:0')
        a['dtype_to'] = torch.half
        a['dtype_from'] = torch.complex128
        b['dtype_to'] = torch.float64
        b['dtype_from'] = torch.float16
        a['torch2trt_enabled'] = True
        b['output_size'] = [10]
        t1 = torch.full([100000000], 2, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = t1.to(dtype=a['dtype'])
        t3 = torch.logit(t2)
        t4 = torch.cumsum(t3, 1)
        t5 = convert_element_type(t4, dtype=b['dtype_to'])
        t5 = torch.reshape(t5, b['output_size'])
        t6 = torch.matmul(x2, x3)
        t7 = torch.add(t5, t6)
        return t7
# Inputs to the model
x2 = torch.randn(4, 4, device='cuda:0')
x3 = torch.randn(4, 4, device='cuda:0')
