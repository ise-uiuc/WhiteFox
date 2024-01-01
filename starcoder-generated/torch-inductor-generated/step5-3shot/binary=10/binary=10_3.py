
class Model(torch.nn.Module):
    def __init__(self, w, b):
        super().__init__()
        self.linear = torch.nn.Linear(3, 1)
        self.linear.weight.data = w
        self.linear.bias.data = b

    def forward(self, x1):
        v1 = self.linear(x1)
        return v1

# Initializing the model
weights = torch.tensor([[-0.1222, 0.2531, -1.6051]], requires_grad=True)
bias = torch.tensor([-0.5976], requires_grad=True)
m = Model(weights, bias)

# Inputs to the model
x1 = torch.randn(1, 3)
