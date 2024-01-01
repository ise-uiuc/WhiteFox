
class Model(torch.nn.Module):
    def __init__(self, n_inp, n_features):
        super().__init__()
        self.linear1 = torch.nn.Linear(n_inp, n_features)

    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3

# Initializing model
m = Model(5, 3)

# Inputs to the model
x1 = torch.randn(1, 5)
