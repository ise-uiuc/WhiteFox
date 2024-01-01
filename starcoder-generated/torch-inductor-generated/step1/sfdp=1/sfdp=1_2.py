
class Model(torch.nn.Module):
    def __init__(self, n_features, n_neurons, p):
        super().__init__()
        self.query = torch.nn.Linear(n_features, n_neurons)
        self.key = torch.nn.Linear(n_features, n_neurons)
        self.value = torch.nn.Linear(n_features, n_neurons)
        self.p = p
 
    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        numerator = (query @ key.transpose(-2, -1))
        v1 = numerator / math.sqrt(query.size(-1))
        v1 = v1.softmax(dim=-1)
        v2 = torch.nn.functional.dropout(v1, p=self.p, training=self.training)
        v3 = v2 @ value
        return v3

# Initializing the model
m = Model(100, 200, 0.2)

# Inputs to the model
x = torch.randn(30, 5, 100)
