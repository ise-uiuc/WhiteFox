
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transp = torch.nn.ConvTranspose2d(3, 32, kernel_size=(1, 15), stride=(2, 15), padding=(1, 0), output_padding=(0, 1))
    def forward(self, x1):
        v1 = self.conv_transp(x1)
        v2 = torch.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 112, 224)
