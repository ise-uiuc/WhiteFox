
class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(224, 8, bias=True)

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = relu(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 224)
