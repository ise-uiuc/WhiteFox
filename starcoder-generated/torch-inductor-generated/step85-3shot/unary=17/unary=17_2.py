
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        return
    def forward(self, x1):
        v1 = torch.randn(1, 1, 80, 80)
        v2 = torch.relu(v1)
        v3 = torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
