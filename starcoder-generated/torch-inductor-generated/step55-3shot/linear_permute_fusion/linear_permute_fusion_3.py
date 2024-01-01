
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3).cuda()
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias).cuda()
        v3 = x1.cuda()
        v2 = v1.permute(0, 2, 1)
        return v3.permute(1, 0, 2) + v2.flip(0)
# Inputs to the model
x1 = torch.randn(2, 3, 3)
