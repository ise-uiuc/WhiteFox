
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.sigmoid = torch.nn.Sigmoid()
        self.max = torch.nn.MaxPool2d((1, 1))
        self.avg_pool2d = torch.nn.AvgPool2d((1, 1))
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = (
            torch.nn.functional.conv2d(v1, self.linear.weight, self.linear.bias)
          .permute(0, 2, 3, 1)
          .reshape(3, 1)
        )

        x = self.sigmoid(x1 + v1)
        x1 = self.max(x)
        x2 = self.avg_pool2d(x1)
        return (x2 + x) * x * v1 ** 2 * x1
# Inputs to the model
x1 = torch.randn(1, 4, 4)
