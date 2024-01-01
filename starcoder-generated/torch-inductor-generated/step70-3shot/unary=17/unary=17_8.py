
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(3, 3, 4, stride=4, dilation=1, padding=0)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(3, 3, 3, stride=3, dilation=1, padding=1)
        self.conv_transpose3 = torch.nn.ConvTranspose2d(3, 3, 2, stride=2, dilation=1, padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v1 = self.relu(v1)
        v2 = self.conv_transpose2(v1)
        v2 = self.relu(v2)
        v3 = self.conv_transpose3(v2)
        v3 = self.relu(v3)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
