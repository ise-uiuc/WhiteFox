
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_functional = torch.nn.functional.relu
        self.linear = torch.nn.Linear(2, 2).cuda()
    def forward(self, x1):
        v1 = self.linear_relu_functional(self.linear(x1))
        v2 = v1.permute(0, 2, 1).cuda()
        v3 = v2.permute(0, 2, 1).cuda()
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 2, device='cuda')
