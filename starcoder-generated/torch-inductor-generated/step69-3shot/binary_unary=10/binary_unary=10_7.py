
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = torch.nn.Linear(4, 7, bias=True)
        self.relu_layer = torch.nn.ReLU()
        self.other_tensor = torch.rand(1, 4)

    def forward(self, x1):
        v1 = self.linear_layer(x1)
        v2 = v1 + self.other_tensor
        v3 = self.relu_layer(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4)
