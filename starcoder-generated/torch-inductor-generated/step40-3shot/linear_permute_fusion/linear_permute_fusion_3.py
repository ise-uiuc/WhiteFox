
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        v1 = self.relu(x1)
        v3 = v1
        v2 = v3.permute(0, 2, 3, 1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 2, 2)
