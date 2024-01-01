
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 16)

    def forward(self, input_tensor, other):
        v1 = self.linear(input_tensor)
        v2 = v1 + other
        return v2

# Initializing the model
m = Model()

# Input tensors to the model
x1 = torch.randn(1, 10)
x2 = torch.randn(1, 16)

