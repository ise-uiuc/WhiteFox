
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2).cuda()
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        a1 = v1.cuda()
        v2 = a1.permute(0, 2, 1)
        return (v1, v2)
# Inputs to the model
x1 = torch.randn(3, 2, 2, device='cuda')
