
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(3, 10, 3, padding=1, stride=2)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = torch.relu(v1)
        v3 = v2.transpose(2, 1)
        v4 = v3.flatten(1)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 20, 20)
