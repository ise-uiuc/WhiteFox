
class Model(torch.nn.Module): 
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1, x2):
        v1 = x2.permute(1, 2, 0)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        x3 = torch.nn.functional.gelu(v2).clone()
        x4 = x1.permute(1, 2, 0).clone()
        v3 = torch.max(x3+x4)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 3)
x2 = torch.randn(5, 3, 4)
