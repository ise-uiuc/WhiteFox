
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
    def forward(self, x1):
        v1 = self.relu(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 1, 1)
