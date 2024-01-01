
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2, bias=False)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, bias=None)
        a1 = torch.flatten(v1, 1, -1).cuda()
        v2 = torch.nn.functional.relu(a1)
        return (v1, v2)
# Inputs to the model
x1 = torch.randn(1, 2, 2).cuda()
