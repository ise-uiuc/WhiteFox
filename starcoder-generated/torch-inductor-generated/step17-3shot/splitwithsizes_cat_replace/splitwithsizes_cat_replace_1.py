
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 32, 3, 1, 1), torch.nn.Conv2d(32, 32, 3, 1, 0))
    def forward(self, x1):
        concatenated_tensor = torch.cat([self.features(x1), self.features(x1)], dim=2)
        return concatenated_tensor
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
