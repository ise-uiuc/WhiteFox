
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(5, 16, 7, padding=3)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.relu(v1)
        v3 = torch.sigmoid(v2)
        v4 = torch.tanh(v3)
        v5 = torch.tanh(v3)
        v6 = torch.tanh(v5)
        v7 = torch.flatten(v6, start_dim=1)
        return v7
# Inputs to the model
x1 = torch.randn(1, 5, 8, 8)
