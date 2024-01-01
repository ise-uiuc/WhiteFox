
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.avg_pool2d = torch.nn.AvgPool2d(kernel_size=(1, 1), stride=(1, 1), padding=(0,), ceil_mode=False, count_include_pad=True)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = v2.unsqueeze(1).unsqueeze(2)
        v4 = self.avg_pool2d(v3)
        v5 = v4.squeeze()
        v6 = v2 + v5
        return v6
# Inputs to the model
x1 = torch.randn(1, 2, 2)
