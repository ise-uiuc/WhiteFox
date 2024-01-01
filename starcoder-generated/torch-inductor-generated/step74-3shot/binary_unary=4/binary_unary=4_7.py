
class Model(torch.nn.Module):
    def forward(self, x1, another):
        v1 = torch.nn.functional.linear(input=x1, weight=self.weight, bias=self.bias)
        v2 = v1 + another
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
m = Model()

# Input to the model
x4 = torch.randn(3, 8, 128)
m = Model()
m.weight = torch.nn.Parameter(torch.randn(10, 8))
m.bias = torch.nn.Parameter(torch.randn(10))

v1 = torch.nn.functional.linear(input=x4, weight=self.weight, bias=self.bias)
v2 = v1 + 5
v3 = torch.nn.functional.relu(v2)
