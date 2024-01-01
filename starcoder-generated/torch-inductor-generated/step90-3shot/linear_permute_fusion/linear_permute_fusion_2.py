
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 2).cuda()
    def forward(self, x5):
        v5 = torch.nn.functional.linear(x5, self.linear.weight, self.linear.bias)
        v6 = v5.permute(2, 1, 0).cuda()
        v7 = v6.transpose(-1, -2).cuda()
        return v6
# Inputs to the model
x5 = torch.randn(3, 2, 2, device='cuda')
