
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2).cuda()
    def forward(self, x1):
        a1 = torch.tanh(x1)
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v4 = torch.reshape(a1, (1, 4))
        v2 = v1.permute(0, 2, 1).cuda()
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 2, device='cuda')
