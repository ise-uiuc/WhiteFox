
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(torch.nn.ConvTranspose2d(3, 32, 3, bias=True, padding=1), torch.nn.ReLU(inplace=False), torch.nn.ConvTranspose2d(32, 3, 3, bias=True, padding=1), torch.nn.ReLU(inplace=False), torch.nn.Sigmoid())
    def forward(self, x1):
        y = self.model(x1)
        return y
# Inputs to the model
x1 = torch.randn(1, 3, 100, 100)
