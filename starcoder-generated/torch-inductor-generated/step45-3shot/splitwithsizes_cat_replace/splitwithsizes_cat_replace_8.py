
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        block_0 = [torch.nn.Conv2d(3, 32, 3, 1, 1, bias=False)]
        block_1 = [torch.nn.ReLU()]
        block_2 = [torch.nn.Conv2d(32, 64, 3, 2, 1, bias=False), torch.nn.BatchNorm2d(64, affine=False, track_running_stats=False)]
        block_3 = [torch.nn.ReLU()]
        block_4 = [torch.nn.Conv2d(32, 64, 1, 1, 1, bias=False)]
        self.features = torch.nn.Sequential(*block_0, *block_1, *block_2, *block_3, *block_4)
    def forward(self, v1):
        x = self.features(v1)
        return (x)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
