
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(4, 4, (3, 3), (1, 1))
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(4, 4)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.flatten(v1)
        v3 = self.linear(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 4, 120, 120)
