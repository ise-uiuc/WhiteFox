
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 32, 3, 2)
    def forward(self, x1):
        v5 = self.conv_transpose(x1)
        v6 = 0.01*v5.transpose(3, 2)
        v7 = torch.sigmoid(v6)
        v8 = v7.transpose(3, 2)
        v9 = torch.relu(v8)
        v10 = v9.transpose(3, 2)
        v11 = 1.1*v10
        v12 = 1.3*v11
        v13 = -0.1*v12
        v14 = v13.transpose(3, 2)
        return v14
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
