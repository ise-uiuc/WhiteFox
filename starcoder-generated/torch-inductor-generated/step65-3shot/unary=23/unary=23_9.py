
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 2, kernel_size=(3, 1))
        self.linear = torch.nn.Linear(3, 2)
    def forward(self, x1, x2):
        v1 = self.conv_transpose(x1)
        v2 = torch.tanh(v1)
        v3 = torch.matmul(v2, x2)
        v4 = self.linear(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 16, 16)
x2 = torch.randn(2, 3)
