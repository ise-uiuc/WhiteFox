
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = torch.nn.Sequential(torch.nn.ConvTranspose2d(3, 16, 3, padding=1, stride=2, output_padding=0, bias=True), torch.nn.ReLU(inplace=False))
        self.block1 = torch.nn.Sequential(torch.nn.ConvTranspose2d(16, 32, 3, padding=1, stride=2, output_padding=0, bias=True), torch.nn.ReLU(inplace=False))
        self.block2 = torch.nn.Sequential(torch.nn.ConvTranspose2d(32, 3, 3, padding=1, stride=2, output_padding=0, bias=True), torch.nn.ReLU(inplace=False))
    def forward(self, x1):
        v1 = self.block0(x1)
        v2 = self.block1(v1)
        v3 = self.block2(v2)
        return torch.squeeze(v3, dim=0)
# Inputs to the model
x1 = torch.randn(1, 3, 128, 32)
