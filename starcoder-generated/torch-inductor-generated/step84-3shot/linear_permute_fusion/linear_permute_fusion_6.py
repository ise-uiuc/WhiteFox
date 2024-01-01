
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4, bias=True)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        a1 = v1.permute(0, 2, 1)
        softmax1 = torch.nn.functional.softmax
        a2 = softmax1(a1, dim=-1)
        v2 = a2.unsqueeze(2)
        return v2
# Inputs to the model
x1 = torch.randn(3, 2, 2)
