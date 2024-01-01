
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layers = [torch.nn.Sequential(
            torch.nn.Conv2d(59, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )]
    def forward(self, x1):
        v1 = torch.nn.functional.relu(self.layers[0](x1))
        v2 = v1 + 3
        v3 = torch.clamp(v2, min=0)
        v4 = torch.clamp(v3, max=6)
        v5 = v1 * v4
        v6 = v5 / 6
        return v6
# Inputs to the model
x1 = torch.randn(1, 59, 30, 40)
