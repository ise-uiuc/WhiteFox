
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(10, 4, 3, padding=1, stride=2)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(6, 2, 3, padding=1, stride=2)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = torch.relu(v1)
        v3 = v2
        v4 = torch.softmax(v2, dim=-1)
        v5 = v4
        v6 = self.conv_transpose2(v5)
        v7 = torch.sigmoid(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 10, 13, 13)
