
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
 
    def forward(self, x1, other):
        v1 = self.linear(x1)
        v2 = v1 + other
        v3 = torch.relu(v2)
        return v3

# Initializing the model
def make_model(n_features, n_labels):
    return Model()

# Inputs to the model
x1 = torch.randn(20, 8)
other = torch.randn(20, 8)
