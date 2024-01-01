
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.ModuleList([torch.nn.Conv2d(32, 32, 3, 1, 1)])
    def forward(self, v1):
        concatenated_tensor = torch.cat(torch.split(v1, [1, 1, 1, 1], dim=1), dim=1)
        return concatenated_tensor
# Inputs to the model
x1 = torch.randn(1, 32, 64, 64)
