
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = nn.ConvTranspose2d(49, 9, 3, stride=1, padding=1)
        self.conv_transpose2 = nn.ConvTranspose2d(9, 6, 3, stride=2, padding=1, output_padding=(1, 1))
    def forward(self, x):
        x = self.conv_transpose1(x)
        x = x * 0.5
        x = x * x * x
        x = x * 0.044715
        x = x + self.conv_transpose2(x)
        x = x * 0.7978845608028654
        x = torch.tanh(x)
        x = x + 1
        x = x * x
        return x
# Inputs to the model
x1 = torch.randn((1, 49, 2, 5))
