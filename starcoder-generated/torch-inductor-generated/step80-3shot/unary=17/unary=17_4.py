
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convt_block = torch.nn.Sequential(torch.nn.ConvTranspose2d(512, 256, kernel_size=3, padding=0, stride=2, output_padding=1), torch.nn.ReLU(inplace=True))
    def forward(self, x1):
        v1 = self.convt_block(x1)
# Inputs to the model
x1 = torch.randn(1, 512, 4, 4)
