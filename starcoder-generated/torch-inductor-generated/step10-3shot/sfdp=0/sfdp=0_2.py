
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Sequential(
                    torch.nn.Sequential(
                        torch.nn.BatchNorm2d(3, eps=1e-08, momentum=0.1, affine=True, track_running_stats=True),
                        torch.nn.Conv2d(3, 74, (1, 1), stride=(1, 1), bias=False),
                     ),
                     torch.nn.Sequential(
                        torch.nn.BatchNorm2d(74, eps=1e-08, momentum=0.1, affine=True, track_running_stats=True),
                       torch.nn.Conv2d(74, 74, (3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                     )
                )
        self.layer2 = torch.nn.Sequential(
                    torch.nn.Sequential(
                        torch.nn.BatchNorm2d(74, eps=1e-08, momentum=0.1, affine=True, track_running_stats=True),
                        torch.nn.Conv2d(74, 74, (1, 1), stride=(1, 1), bias=False),
                     ),
                     torch.nn.Sequential(
                        torch.nn.BatchNorm2d(74, eps=1e-08, momentum=0.1, affine=True, track_running_stats=True),
                       torch.nn.Conv2d(74, 1, (3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                     ),
)

    def forward(self, x1):
        x = self.layer1(x1)
        x = self.layer2(x).squeeze()
        return x
# Inputs to the model
x1 = torch.randn(5, 3, 224, 224)
