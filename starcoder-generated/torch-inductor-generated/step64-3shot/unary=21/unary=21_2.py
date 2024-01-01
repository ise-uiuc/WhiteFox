
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ConvNormRelu = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=(3, 3), padding=(1, 1)),
            torch.nn.LayerNorm([7, 7, 64]),
            torch.nn.LeakyReLU(negative_slope=0.1)
        )
        self.ConvTranspose2d2 = torch.nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.Tanh3 = torch.nn.Tanh()
        self.ConvTranspose2d4 = torch.nn.ConvTranspose2d(64, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.Sigmoid5 = torch.nn.Sigmoid()
    def forward(self, x):
        v1 = self.ConvNormRelu(x)
        v2 = self.ConvTranspose2d2(v1)
        v3 = torch.tanh(v2)
        v4 = self.ConvTranspose2d4(v3)
        v5 = self.Sigmoid5(v4)
        return v5
# Inputs to the model
x = torch.randn(1, 3, 28, 28)
