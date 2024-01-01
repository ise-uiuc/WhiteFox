
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 3, kernel_size=(2, 3), stride=(2, 1))
        self.soft_max = torch.nn.Softmax(dim=2)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.relu(v1)
        v3 = torch.tanh(v2)
        v4 = torch.mean(v3)
        v5 = torch.floor(v4)
        v6 = torch.sum(v3)
        v7 = self.soft_max(v3)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 16, 25)
