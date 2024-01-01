
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2, bias=False).cuda()
    def forward(self, x1):
        v4 = x1
        v3 = v4.view(1)
        v1 = torch.nn.functional.linear(v3, self.linear.weight)
        v2 = v1.permute(0, 2, 1).cuda()
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
