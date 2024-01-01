
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = x1[:, 0].norm() - x2[0].norm() + x1[:, 1].norm() - x2[1].norm()
        return torch.cat([v1, v1], 1)
        
# Inputs to the model
x1 = torch.randn(2, 1)
x2 = torch.randn(2, 1)
