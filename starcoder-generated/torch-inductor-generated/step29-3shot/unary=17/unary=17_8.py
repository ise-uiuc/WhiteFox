
class A(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_0 = nn.ConvTranspose2d(3, 2, 3, padding=1, stride=2)
        self.conv_1 = nn.ConvTranspose2d(2, 3, 3, padding=1, stride=2)
    def forward(self, x):
        y1 = F.relu(self.conv_0(x))
        output = F.relu(self.conv_1(y))
        return output
# Inputs to the model
x_1 = torch.randn(1, 3, 32, 32)

