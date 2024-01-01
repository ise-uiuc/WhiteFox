
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size=(8,8), stride=(10,10))
        self.linear = torch.nn.Linear(224*224, 4096)
    def forward(self, x1):
        v1 = self.avgpool(x1)
        v2 = v1.view((-1, self.linear.in_features))
        v3 = self.linear(v2)
        v4 = v3 * 0.5
        v5 = v3 * 0.7071068
        v7 = v5.mean((-1))
        v8 = v4 + v7
        v9 = v8.reshape((-1, 3, 32, 32))
        return v9
# Inputs to the model
x1 = torch.randn(1, 7, 100, 100)
