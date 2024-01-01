
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = x1.squeeze(-1)
        v2 = v1 - 0.5
        v3 = F.relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(2, 3, 32, 224, 224)
