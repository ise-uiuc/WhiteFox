
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(3, 6, padding=4)
    def forward(self, x):
        x = torch.relu(x)
        x = self.conv(x)
        x = x - 3.0
        return x
# Inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)
