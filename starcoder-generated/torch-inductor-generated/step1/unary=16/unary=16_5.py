
class Model(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, x):
        y = torch.relu(self.linear(x))
        return y

# Initializing the model
m = Model(16, 32)

