
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 3)
    def forward(self, x0):
        v0 = torch.nn.functional.linear(x0, self.linear.weight, self.linear.bias)
        v1 = v0.permute(0, 2, 1).cuda()
        v2 = v1.contiguous().cuda()
        return v2
# Inputs to the model
x0 = torch.randn(1, 2, 2, device='cuda')
