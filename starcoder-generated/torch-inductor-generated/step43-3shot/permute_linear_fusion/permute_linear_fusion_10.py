
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=1, stride=1, padding=0, ceil_mode=False, count_include_pad=True)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        x2 = v2.unsqueeze(dim=1)
        v3 = self.avg_pool(x2)
        x3 = v3.squeeze(dim=1)
        v4 = x2.squeeze(dim=1)
        v5 = v4 + x3
        return v5
# Inputs to the model
x1 = torch.randn(1, 2, 2)
