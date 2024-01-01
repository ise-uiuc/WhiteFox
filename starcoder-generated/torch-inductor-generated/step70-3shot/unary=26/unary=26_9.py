
class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.r1 = nn.ConvTranspose2d(64, 3, (9, 9), (1, 1), (4, 4), 1, 1, bias=False)
        self.r2 = nn.ConvTranspose2d(3, 32, (3, 3), (2, 2), (0, 0), 1, 1, bias=False)
        self.r3 = nn.ConvTranspose2d(32, 64, (3, 3), (2, 2), (1, 1), 1, 1, bias=False)
        self.r4 = nn.BatchNorm2d(64)
        
    def forward(self, x18):
        x = self.r1(x18.clone())
        x = x > 0
        x = x * 0.603
        x = torch.where(x, x, x)
        x = x.flatten(2)
        x = x.unsqueeze(2)
        x = x.transpose(-1, -2)
        x = self.r2(x.clone())
        x = torch.softmax(x, 1)
        x = torch.clip(x, 0, 1)
        x = x > 0
        x = x * -0.09
        x = torch.where(x, x, x)
        x = x.transpose(-1, -2)
        x = self.r3(x.clone())
        x = nn.ReLU(inplace=True)(x.clone())
        x = nn.Softmax(dim=1)(x.clone())
        x = self.r4(x.clone())
        return x * 0.208

# Inputs to the model
x18 = torch.randn(2, 64, 224, 256, requires_grad=False)
