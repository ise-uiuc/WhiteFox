
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 5, 1, stride=1, padding=1)
        self.conv_transpose1 = torch.nn.ConvTranspose2d(5, 5, 1, stride=1, padding=1)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(5, 5, 1, stride=1, padding=1)
        self.conv_transpose3 = torch.nn.ConvTranspose2d(5, 10, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.conv_transpose1(v1)
        v3 = self.conv_transpose2(v2)
        v4 = self.conv_transpose3(v3)
        v5 = torch.relu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 24, 24)
