
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.mm(x2, x1)
        v2 = torch.mm(x2, x1)
        return torch.cat([v2, v1, v1, v2], dim=1)
# Inputs to the model
x1 = torch.tensor([[0.2, 0.1, -0.3, 0.5]])
x2 = torch.tensor([[0.2, 0.1, 0.9, 0.8]])
