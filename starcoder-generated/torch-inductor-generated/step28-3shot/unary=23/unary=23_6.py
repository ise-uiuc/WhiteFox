
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_transpose = torch.nn.ConvTranspose2d(4, 3, kernel_size=5, padding='valid')
    def forward(self, x1):
        v1 = self.conv1_transpose(x1)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 4, 20, 20)
