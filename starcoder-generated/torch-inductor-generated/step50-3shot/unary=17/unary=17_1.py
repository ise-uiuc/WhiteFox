
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(3, 5, 1, stride=2)
        self.avg_pool2d = torch.nn.AvgPool2d(kernel_size=2)
    def forward(self, x0):
        v0 = self.conv2d(x0)
        v1 = torch.relu(v0)
        v2 = self.avg_pool2d(v1)
        v3 = v2.view((1, -1))
        v4 = torch.softmax(v3)
        v5 = v4.mm(torch.Tensor([[5.18, -18.76, 13.96, 32.95, 43.09, -29.25, -45.07, -12.60, -23.83, 8.78, 53.87, 38.49, 96.30, 44.88, -66.93, -67.88, -51.33]]))
        return v5

# Inputs to the model
x0 = torch.randn(1, 3, 10, 10)
