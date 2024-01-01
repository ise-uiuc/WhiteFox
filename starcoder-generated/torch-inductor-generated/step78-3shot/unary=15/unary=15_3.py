
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 3, (4, 4)), torch.nn.ReLU())
    def forward(self, x1):
        v0 = x1
        v1 = self.features(v0)
        v2 = v1.view(-1)
        v3 = torch.relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 2, 2)
