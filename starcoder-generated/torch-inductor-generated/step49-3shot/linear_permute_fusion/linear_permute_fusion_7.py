
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 3)
    def forward(self, x1):
        t1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        t2 = torch.sum(t1, dim=-1, keepdim=True)
        return t2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
