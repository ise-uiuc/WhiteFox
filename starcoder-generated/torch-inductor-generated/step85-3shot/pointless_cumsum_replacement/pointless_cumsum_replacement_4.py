
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        a = {}
        a = {}
        a['dtype'] = torch.float64
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:0')
        a['dtype_to'] = torch.complex128
        t1 = torch.ones([16, 256, 20, 20], dtype=torch.float64, layout=torch.strided, device=torch.device('cuda:0'))
        tensor_split0, tensor_split1, tensor_split2 = torch.split(t1, 2, 1)
        tensor_view0 = tensor_split0.contiguous().view(16 * 2, 100)
        tensor_view1 = tensor_split1.contiguous().view(16 * 2, 100)
        tensor_view2 = tensor_split2.contiguous().view(16 * 2, 100)
        tensor_cat0 = torch.cat([tensor_view0, tensor_view1, tensor_view2], 0)
        t2 = tensor_cat0.contiguous().view(16, 256 * 100).to(torch.complex128)
        t3 = torch.max(t2, 1)
        return t3[0].add(t3[1]).to(a['dtype'])
# Inputs to the model
x = torch.randn(16, 256, 20, 20, device='cuda:0')
