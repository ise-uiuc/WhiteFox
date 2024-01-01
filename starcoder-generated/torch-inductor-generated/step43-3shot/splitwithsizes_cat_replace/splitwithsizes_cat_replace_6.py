
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.ModuleList([torch.nn.BatchNorm2d(32)])
    def forward(self, v1):
        v2 = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(v2, dim=1)
        return (concatenated_tensor, v2)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
