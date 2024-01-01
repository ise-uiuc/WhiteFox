
torch.nn.functional.conv2d = lambda x, y: x
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.functional.conv2d
    def forward(self, x):
        x = self.layers(x, x)
        x = torch.cat((x, x), dim=3)
        return x
# Inputs to the model
x = torch.randn(1, 4, 4, 4)
