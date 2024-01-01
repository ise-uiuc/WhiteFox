
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 16).cuda()
        self.linear2 = torch.nn.Linear(16, 64).cuda()
        self.linear3 = torch.nn.Linear(64, 112).cuda()
    def forward(self, x0):
        v1 = torch.nn.functional.linear(x0, self.linear1.weight, self.linear1.bias)
        v2 = v1.view(1, 32, 3)
        v3 = torch.nn.functional.linear(v2, self.linear2.weight, self.linear2.bias)
        v5 = torch.nn.functional.linear(v3, self.linear3.weight, self.linear3.bias)
        v4 = v5.view(1, 4, 6, 3)
        v4[v4 < 0] = 0
        v4[v4 > 0] = 1
        return v4.to('cpu')
# Inputs to the model
x0 = torch.randn(1, 4, 6, 2, device='cuda')
