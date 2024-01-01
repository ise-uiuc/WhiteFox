
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t = torch.nn.ConvTranspose2d(in_channels=3, out_channels=3, bias=False, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
        self.relu = torch.nn.ReLU(inplace=True)
    def forward(self, x1):
        v1 = self.t(x1)
        v2 = self.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 224, 256)
