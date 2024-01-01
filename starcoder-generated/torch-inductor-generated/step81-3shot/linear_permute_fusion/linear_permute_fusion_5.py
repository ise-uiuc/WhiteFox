
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(16, 32).cuda()
        self.linear2 = torch.nn.Linear(32, 16).cuda()
        self.linear3 = torch.nn.Linear(16, 4).cuda()
    def forward(self, x0):
        v0 = torch.nn.functional.linear(x0, self.linear1.weight, self.linear1.bias)
        v1 = v0[:, -1]
        v1 = v1.unsqueeze(1).repeat(1, 32)
        v2 = torch.nn.functional.linear(v1, self.linear2.weight, self.linear2.bias)
        v3 = v2[:, -1]
        v3 = v3.unsqueeze(1).repeat(1, 4)
        v4 = torch.nn.functional.linear(v3, self.linear3.weight, self.linear3.bias)
        return v2
# Inputs to the model
x0 = torch.randn(1, 5, 16, device='cuda')
