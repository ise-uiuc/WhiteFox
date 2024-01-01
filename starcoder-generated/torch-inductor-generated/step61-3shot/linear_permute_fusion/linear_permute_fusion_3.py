
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
    def forward(self, x3):
        v3 = torch.nn.functional.linear(x3, self.linear.weight, self.linear.bias)
        flatten2 = torch.flatten(v3, 2)
        reshape2 = flatten2.view([1, 1, 1])
        reshape2_2 = torch.flatten(reshape2, 2)
        return reshape2_2

# Inputs to the model
x3 = torch.randn(1, 1, 3)



