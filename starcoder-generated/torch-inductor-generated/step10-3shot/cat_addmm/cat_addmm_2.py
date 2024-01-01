
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        b = x1.shape[0]
        c = 2
        m = x1.shape[1]
        v1 = torch.addmm(x1, torch.ones(b, c, m, device='cuda'))
        return torch.cat([v1, v1])

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(3, 2, device='cuda')
