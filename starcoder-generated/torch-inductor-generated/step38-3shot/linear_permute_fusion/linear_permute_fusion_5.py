
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2).cuda()
        self.bias = torch.nn.Parameter([0.1, 0.2])
    def forward(self, x1):
        v4 = x1.cuda()
        v1 = torch.nn.functional.linear(v4, self.linear.weight.cuda(), self.linear.bias.cuda())
        v2 = v1.permute(0, 2, 1).cuda()
        v3 = torch.add(v2, self.bias, alpha=0)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
