
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2).cuda()
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 1, 3, 2)
        v2[(v2 < 0)] = 0
        v2[(v2 > 0)] = 1
        return v2.to('cpu')
# Inputs to the model
x1 = torch.randn(1, 2, 2, 2, device='cuda')
