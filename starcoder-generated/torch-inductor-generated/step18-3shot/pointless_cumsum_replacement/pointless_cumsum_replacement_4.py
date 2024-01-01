
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        b = {}
        a = {}
        a['dtype'] = torch.float64
        a['layout'] = torch.strided
        a['device'] = torch.device('cpu')
        b['dtype'] = torch.float16
        b['layout'] = torch.strided
        b['device'] = torch.device('cpu')
        a['dtype_to'] = torch.float64
        a['dtype_from'] = torch.float64
        b['dtype_to'] = torch.float16
        b['dtype_from'] = torch.float64
        t1 = torch.full([256, 1024], 1.0, dtype=a['dtype'], layout=a['layout'], device=a['device'], pin_memory=False)
        t2 = t1.to(dtype=b['dtype'])
        t3 = t2.to(dtype=a['dtype'])
        t4 = torch.add(t3, x1)
        t5 = torch.mul(t4, x2)
        t6 = torch.cumsum(t5, 1)
        return t6


# Inputs to the model
x1 = torch.tensor([-2.6429, 0.0329, 0.94, 0.589, 0.3317,
0.294, 0.6286, -0.3628, 0.9315,
0.6357],
requires_grad=True,
device='cpu')
x2 = torch.tensor([1.0194,
-0.8455,
-0.0254,
0.3436,
0.0239,
-0.4474,
-0.891,
-0.2121,
0.7194,
0.262],
requires_grad=True,
dtype=torch.float64,
device='cpu')

