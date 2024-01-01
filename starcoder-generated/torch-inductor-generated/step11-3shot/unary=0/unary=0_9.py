
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.relu(x1)
        v2 = torch.relu(x1)
        v3 = torch.relu(x1)
        v4 = torch.relu(x1)
        v5 = torch.relu(x1)
        v6 = torch.relu(x1)
        v7 = torch.relu(x1)
        v8 = torch.relu(x1)
        return v8
# Inputs to the model
x1 = torch.randn(1, 8, 128, 90)
