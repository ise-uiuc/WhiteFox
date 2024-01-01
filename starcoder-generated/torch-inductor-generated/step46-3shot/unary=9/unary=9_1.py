
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # 3x3 conv->batchnorm->activation->3x3 conv
        self.conv_bn_act_1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, 3, stride=1, padding=1),
        )

        # 7x7 conv->batchnorm->activation->3x3 conv
        self.conv_bn_act_2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 12, 7, stride=1, padding=3),
            torch.nn.BatchNorm2d(12),
            torch.nn.ReLU(),
            torch.nn.Conv2d(12, 12, 3, stride=1, padding=1),
        )

    def forward(self, x1):
        t1 = self.conv_bn_act_1(x1)
        t2 = self.conv_bn_act_2(t1)
        return t2
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
