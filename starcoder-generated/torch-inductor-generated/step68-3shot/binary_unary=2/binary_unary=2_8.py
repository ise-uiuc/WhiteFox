
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.tanh(F.hardtanh(x1))
        v2 = F.hardtanh(v1)
        v3 = F.hardtanh(v2)
        v4 = v3 + v3
        v5 = v1.view(1, 2, 4)
        v6 = v2.unsqueeze(1)
        return torch.relu(v4)
# Inputs to the model
x1 = torch.randn(8, 4)
