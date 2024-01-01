
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(5, 1, 3, padding=1, stride=2)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = torch.relu(v1)
        v3 = torch.sigmoid(v2)
        v4 = torch.transpose(v3, 2, 1)
        v5 = torch.flatten(v4, 1)
        v6 = torch.softmax(v5, dim=1)
        return v6
# Inputs to the model
x1 = torch.randn(1, 5, 20, 20)
