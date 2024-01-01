
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2).to('cuda')
    def forward(self, x1, x2):
        v0 = 2 * x1 - x2
        v1 = torch.nn.functional.linear(v0, self.linear.weight, self.linear.bias)
        v2 = v1.transpose(0, 1)
        return v2
# Inputs to the model
x1 = torch.randn(2, 3, device='cuda')
x2 = torch.randn(2, device='cuda')
