
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
    def forward(self, x4):
        v4 = torch.nn.functional.linear(x4, self.linear.weight, self.linear.bias)
        v5 = v4.permute(0, 2, 1).cuda()
        v6 = v5.permute(0, 2, 1)
        return v5
# Inputs to the model
x4 = torch.randn(2, 3, 3, device='cuda')
