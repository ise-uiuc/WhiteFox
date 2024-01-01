
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 1, (1, 1), stride=(1, 1), padding=(0, 0), output_padding=(0, 0), bias=None, dilation=(1, 1))
        
    def forward(self, x1):
        x4 = self.conv_transpose(x1)
        x5 = torch.tanh(x4)
        return x5
# Inputs to the model
x1 = torch.randn(1, 1, 2, 1)
