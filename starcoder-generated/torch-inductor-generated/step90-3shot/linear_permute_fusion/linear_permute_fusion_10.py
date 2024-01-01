
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2).cuda()
    def forward(self, x5):
        v5 = torch.nn.functional.linear(x5, self.linear.weight, self.linear.bias)
        v6 = v5.permute(0, 2, 1).cuda()
        v7 = v6.permute(0, 2, 1).cuda()
        return v5
# Inputs to the model
x5 = torch.randn(2, 3, 2, device='cuda')
