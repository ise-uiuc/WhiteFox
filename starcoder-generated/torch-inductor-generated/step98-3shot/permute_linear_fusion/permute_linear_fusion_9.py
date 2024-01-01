
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.conv = torch.nn.Conv2d(4, 1, 3, stride=(2), padding=(1))
        self.max_pooling = torch.nn.MaxPool2d(kernel_size=(2), stride=(2))
        self.avg_pooling = torch.nn.AvgPool2d(kernel_size=(2), stride=(2))
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = x1.permute(0, 3, 1, 2)
        v4 = self.conv(v3)
        v5 = self.max_pooling(v4)
        v6 = self.avg_pooling(x1)
        return v3[0, 0, 0, 0] + v1[0, 0, 0] + v5[0, 0, 0, 0] + v4[0, 0, 0, 0]
# Inputs to the model
x1 = torch.randn(1, 4, 2, 2)
