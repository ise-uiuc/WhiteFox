
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m0 = torch.nn.Conv2d(51, 72, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.m1 = torch.nn.BatchNorm2d(72, eps=1.1754943508222875e-38, momentum=0.800000011920929)
        self.m2 = torch.nn.ConvTranspose2d(72, 72, 1, bias=False)

    def forward(self, x7):
        x8 = self.m0(x7)
        x9 = torch.nn.functional.relu_(x8)
        x13 = torch.nn.functional.max_pool2d(x9, 2, 2, 0)
        x12 = self.m1(x13)
        x14 = self.m2(x12)
        x10 = torch.nn.functional.relu_(x12)

        x10 = torch.clamp(x10, min=0)
        x10 = torch.nn.functional.adaptive_avg_pool2d(x10, (1, 1))

        return x10
# Inputs to the model
x7 = torch.randn(1, 51, 97, 12)
