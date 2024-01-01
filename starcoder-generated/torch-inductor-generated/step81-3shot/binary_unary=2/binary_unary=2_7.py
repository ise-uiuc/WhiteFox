
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(45, 32)
    def forward(self, x1):
        v1 = torch.reshape(x1, (-1, 45))
        v2 = self.fc(v1)
        v3 = v2 - torch.randn(1, 32)
        v4 = F.relu(v3)
        v5 = torch.tanh(v4)
        v6 = v5 + torch.abs(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 14, 14)
