
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convTranspose = torch.nn.ConvTranspose2d(2, 2, kernel_size=2, stride=2, padding=1, output_padding=1)
    def forward(self, x):
        