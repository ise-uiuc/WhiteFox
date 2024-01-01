
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 64)
    def forward(self, x1):
        v1 = F.relu(x1)
        v2 = self.linear(v1.flatten(start_dim=1))
        v3 = v2.clamp(-1, 1)
        return torch.sum(v3)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
