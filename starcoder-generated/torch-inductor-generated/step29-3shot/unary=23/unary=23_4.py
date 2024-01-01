
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(2, 2, 2, stride=1)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(2, 9, 3, stride=6)
    def forward(self, x):
        r1 = self.conv_transpose1(x)
        r2 = torch.tanh(r1)
        r3 = self.conv_transpose2(r2)
        return r3
# Inputs to the model
x1 = torch.Tensor(1, 2, 2, 2)
