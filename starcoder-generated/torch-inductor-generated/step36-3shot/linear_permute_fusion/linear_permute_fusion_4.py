
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2).cuda()
        self.relu = torch.nn.ReLU().cuda()
    def forward(self, x1):
        v1, v3 = x1
        a0 = torch.nn.functional.linear(v3, self.linear.weight, self.linear.bias)
        v2 = a0.permute(0, 2, 1)
        a1 = self.relu(v2)
        v4 = (a1, a0)
        return v4
# Inputs to the model
x1 = (torch.randn(2, 2, 2, device='cuda'), torch.randn(2, 2, 2, device='cuda'))
