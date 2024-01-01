
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.relu(x1) + 1
        v2 = torch.sigmoid(v1) + 2
        v3 = torch.tanh(v2) + 3
        v4 = torch.relu(v3) + 4
        return v4
# Inputs to the model
x = torch.randn(1, 2, 3, 4)
