
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.conv1 = torch.nn.Conv2d(3, 3, 3)
        torch.manual_seed(2)
        self.conv2 = torch.nn.Conv2d(3, 3, 3)
        torch.manual_seed(3)
        bn1_weight = torch.randn(1)
        bn2_weight = torch.randn(1)
    def forward(self, x1):
        y1 = self.conv1(x1)
        y2 = self.conv2(x1)
        x2[0] = y1[0] + y2[0]
        y1 = torch.randn(2)
        return y1
# Inputs to the model
x1 = torch.randn(3, 3, 4, 4)
