
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 2, 1, stride=1, padding=1)
    def forward(self, x_in):
        y_in = self.conv_transpose(x_in)
        y_in = torch.tanh(y_in)
        y_in = torch.transpose(y_in, 1, 2)
        y_in = y_in + torch.transpose(y_in, 1, 2)
        y_in = y_in * 0.3
        return y_in
# Inputs to the model
x_in = torch.randn(4, 1, 17, 15)
