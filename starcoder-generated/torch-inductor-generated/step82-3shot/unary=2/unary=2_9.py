
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 2, 3, stride=2, padding=1, groups=2)
        self.fc = torch.nn.Linear(192, 8)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1.flatten(1, -1)
        v3 = self.fc(v2)
        return v3
# Inputs to the model
x1 = torch.randn(2, 4, 10, 10)
