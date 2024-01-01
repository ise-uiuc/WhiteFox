
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        torch.manual_seed(1)
        self.preproc = nn.Sequential(
            nn.Conv2d(3, 2, 2),
            nn.BatchNorm2d(2, affine=False),
        )
    def forward(self, x3):
        x1 = self.preproc(x3)
        x2 = self.preproc(x1)
        return x1
# Inputs to the model
x3 = torch.randn(1, 3, 224, 224)
