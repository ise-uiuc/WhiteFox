
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(torch.nn.Conv2d(6, 16, 3, stride=1, padding=1), torch.nn.Conv2d(16, 6, 3, stride=1, padding=1), torch.nn.Conv2d(6, 16, 3, stride=1, padding=1))
    def forward(self, x1):
        v1 = self.layers(x1)
        v2 = torch.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 6, 231, 429)
