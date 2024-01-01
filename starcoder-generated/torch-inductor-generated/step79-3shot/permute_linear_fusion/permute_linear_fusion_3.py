
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        t1 = x1.permute(0, 2, 1)
        torch.nn.functional.linear(t1, self.linear.weight, self.linear.bias) # Call the module
        t2 = x1.permute(0, 2, 3, 1)
        torch.nn.functional.linear(t2, self.linear.weight, self.linear.bias) # Call the module
        return x1
# Inputs to the model
x1 = torch.randn(1, 2, 2)
#Model ends

# Model begins
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = v1.permute(0, 1, 3, 2)
        v3 = v2.permute(0, 3, 1, 2)
        return v3
# Inputs to the model
x1 = torch.randn(2, 2, 2, 2)
