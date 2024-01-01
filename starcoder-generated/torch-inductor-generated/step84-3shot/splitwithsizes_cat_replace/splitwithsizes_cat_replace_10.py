
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.features = torch.nn.Sequential(*[torch.nn.Conv2d(3, 32, 3, 1, 1)])
    def forward(self, v1, v2):
        concatenated_tensor = torch.cat([v1, v2], dim=1)
        return torch.nn.Conv2d(33, 16, 3, 1, 1)(concatenated_tensor)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 32, 32)
