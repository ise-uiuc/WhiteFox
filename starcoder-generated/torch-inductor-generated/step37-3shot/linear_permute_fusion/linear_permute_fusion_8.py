
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        #self.t = torch.nn.functional.celu
        self.t = torch.nn.functional.gelu
        self.linear = torch.nn.Linear(2, 1)
    def forward(self, x1):
        v1 = self.t(x1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        return v2
# Inputs to the model
x1 = torch.randn(1, 2)
