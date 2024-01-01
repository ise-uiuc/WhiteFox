
class Model():
    def __init__(self):
        super().__init__()  
        self.module = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1, bias=False),
            torch.nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1, bias=False),
            torch.nn.ConvTranspose2d(24, 24, 3, stride=2, padding=1, output_padding=1, bias=False),
            torch.nn.ConvTranspose2d(16, 35, 3, stride=2, padding=1, output_padding=1)
        )
    def forward(self, x0):
        return self.module(x0)
# Inputs to the model
x0 = torch.randn(2, 25, 16, 16)
