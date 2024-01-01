
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, x2):
        x_noise = torch.rand_like(x2)
        v1 = torch.nn.functional.linear(x2, self.linear.weight, self.linear.bias)
        v2 = torch.nn.functional.dropout(v1, p=0.5)
        v3 = torch.nn.functional.dropout(v2, p=0.5)
        v4 = v1.add(v2).add(v3)
        v4 = x2
        return v4

# Initializing the model
m2 = Model()

torch.manual_seed(0)

# Inputs to the model
x2 = torch.randn(1, 2, 2)
