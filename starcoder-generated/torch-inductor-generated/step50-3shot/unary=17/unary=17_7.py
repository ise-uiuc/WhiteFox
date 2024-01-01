
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = torch.nn.Sequential(torch.nn.ConvTranspose2d(in_channels=3, out_channels=8, kernel_size=(5, 5), stride=(2, 2), padding=2, output_padding=1), torch.nn.ReLU(), torch.nn.Sigmoid())
    def forward(self, x):
        x1 = self.block(x)
        return x1
# Input to the model
x = torch.randn(1, 3, 16, 16)
