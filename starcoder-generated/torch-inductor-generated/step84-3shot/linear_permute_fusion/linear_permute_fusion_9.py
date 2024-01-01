
class Model(torch.nn.Module):
    torch.set_printoptions(precision=None, threshold=float("inf"))
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2).cuda()
        self.softmax = torch.nn.Softmax(dim=1).cuda()
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        v3 = torch.nn.functional.softmax(v2, dim=-1)
        softmax1 = self.softmax
        v4 = softmax1(v3)
        v5 = v4.unsqueeze(2)
        return v5
# Inputs to the model
x1 = torch.randn(1, 2, 2, device='cuda')
