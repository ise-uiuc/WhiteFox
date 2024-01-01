
class Model(torch.nn.Module):
    def __init__(self, out_features):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(192, out_features)

    def forward(self, input1):
        v1 = self.linear(input1)
        v2 = v1 - 0.5
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
m = Model(100)

# Inputs to the model
x1 = torch.randn(1, 192)
