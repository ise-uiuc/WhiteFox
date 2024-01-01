
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(128, 128, 2, stride=(1, 1), padding=(2, 2), output_padding=(1, 1), bias=False)
        self.relu = torch.nn.ReLU()
    def forward(self, x2):
        x1 = self.conv_t(x2)
        x2 = self.relu(x1)
        return x2
# Inputs to the model
x2 = torch.randn(1, 128, 8, 5)
