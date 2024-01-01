
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2, 8, 7)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.layer_wrapper(v1)
        return v2
    def layer_wrapper(self, x):
        v3 = self.conv1(x)
        return v3
# Inputs to the model
x = torch.randn(1, 2, 64, 64)
