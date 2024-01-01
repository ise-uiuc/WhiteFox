
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8).cuda()
    def forward(self, x1):
        v4 = x1
        v1 = torch.nn.functional.linear(v4, self.linear.weight, self.linear.bias)
        v2 = v1.transpose(1, 2).cuda()
        v3 = v2.view(2, 16)
        return v3
# Inputs to the model
x1 = torch.randn(2, 3, 3).cuda()
