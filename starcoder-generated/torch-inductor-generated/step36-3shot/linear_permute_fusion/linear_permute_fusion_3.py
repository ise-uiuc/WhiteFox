
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2).cuda()
    def forward(self, x1):
        a1 = x1.cuda()
        a2 = x1.cuda()
        v1 = torch.nn.functional.linear(a1, self.linear.weight, self.linear.bias)
        v2 = torch.nn.functional.linear(a2, self.linear.weight, self.linear.bias)
        a3 = v1.permute(2, 1, 0).cuda()
        a4 = v2.permute(2, 1, 0).cuda()
        v3 = a3 + a4
        v4 = torch.nn.functional.relu(v3)
        return (v1, v2)
# Inputs to the model
x1 = torch.randn(3, 2, 2, device='cuda')
