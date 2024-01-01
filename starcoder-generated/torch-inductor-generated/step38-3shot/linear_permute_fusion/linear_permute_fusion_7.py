
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1).cuda()
    def forward(self, x1):
        v4 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias).cuda()
        v2 = v4.permute(1, 2, 0).cuda()
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 2).cuda()
