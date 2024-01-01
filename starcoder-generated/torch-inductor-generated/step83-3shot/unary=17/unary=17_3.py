
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 64, (1, 1), bias=False),
            torch.nn.BatchNorm2d(64),
        )
        self.model2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 64, (1, 1), bias=False),
        )
    def forward(self, x):
        out1 = self.model1(x).flatten(start_dim=1)
        out2 = self.model2(out1).squeeze(-1).squeeze(-1)
        return out2
    
# Inputs to the model
x1 = torch.randn(1, 64, 3, 224)
