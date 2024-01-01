
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2).cuda()
    def forward(self, x1, x2):
        a1 = torch.tanh(x1)
        a2 = torch.tanh(x2)
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = torch.nn.functional.linear(x2, self.linear.weight, self.linear.bias)
        v4 = torch.reshape(a1, (1, 4))
        v5 = torch.reshape(a2, (4, 1))
        v3 = torch.mm(v5, v4.t())
        v6 = torch.mm(v3, v2)
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 2, device='cuda')
x2 = torch.randn(1, 2, 2, device='cuda')
