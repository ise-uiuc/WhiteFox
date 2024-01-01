
class Model(torch.nn.Module):
    def __init__(self, min_value=2.701 + 1e-06, max_value=2.999 - 1e-06):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 5, 5, stride=1, padding=2)
        self.conv_transpose1 = torch.nn.ConvTranspose2d(5, 8, 3, stride=2, padding=1)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(8, 32, 4, stride=2, padding=1)
        self.conv_transpose3 = torch.nn.ConvTranspose2d(32, 128, 5, stride=4, padding=2)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv_transpose1(v1)
        v3 = self.conv_transpose2(v2)
        v4 = self.conv_transpose3(v3)
        v5 = v4.clamp(self.min_value, self.max_value)
        return v5
# Inputs to the model
x = torch.randn(1, 3, 63, 63)
