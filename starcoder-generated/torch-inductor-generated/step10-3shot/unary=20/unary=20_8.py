
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(64, 1, kernel_size=(3, 4), stride=(1, 2))
    def forward(self, x):
        out = self.conv1(x)
        return torch.tanh(out)
# Input to the model
x = torch.randn(64, 64, 40, 60)
