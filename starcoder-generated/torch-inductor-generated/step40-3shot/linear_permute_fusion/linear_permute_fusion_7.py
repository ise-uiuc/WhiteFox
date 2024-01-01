
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2).cuda()
        self.linear1 = torch.nn.Linear(2, 2).cuda()
    def forward(self, x1):
        v4 = x1.cuda()
        v5 = torch.nn.functional.relu(v4)
        v1 = torch.nn.functional.linear(v4, self.linear.weight, self.linear.bias)
        v2 = torch.nn.functional.relu6(v1)
        v3 = v2.permute(0, 1, 3, 2)
        v6 = torch.min(v3, dim=2, keepdim=False, out=None).values
        v7 = torch.nn.functional.linear(v5.permute(0, 2, 1), self.linear1.weight, self.linear1.bias)
        return v6
# Inputs to the model
x1 = torch.randn(1, 2, 2, device='cuda')
