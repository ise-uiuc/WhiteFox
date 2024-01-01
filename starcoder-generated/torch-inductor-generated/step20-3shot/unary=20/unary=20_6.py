
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(16, 16, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), output_padding=(0, 0))
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
inp = torch.randn(1, 16, 256, 128)
# Model Ends 
