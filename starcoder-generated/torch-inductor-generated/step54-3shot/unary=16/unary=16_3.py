
class Model(torch.nn.Module):
    def __init__(self, n_inputs, n_hidden, n_classes):
        super().__init__()
        self.linear = torch.nn.Linear(n_inputs, n_hidden)
        self.relu = torch.nn.ReLU()
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = self.relu(v1)
        return v2

# Initializing the model
m = Model(n_inputs=120, n_hidden=100, n_classes=30)

# Inputs to the model
x1 = torch.randn(1, 120)
