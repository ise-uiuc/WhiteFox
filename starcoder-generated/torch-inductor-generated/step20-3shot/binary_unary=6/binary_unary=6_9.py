
class Model(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, inputs):
        linear_out = self.linear(inputs)
        out = linear_out - 2.0
        # For the purpose of being different from the previously generated model.
        out += linear_out
        out = nn.functional.relu(out)
        return out

# Initializing the model
m = Model(in_features=64, out_features=16)

# Inputs to the model
x = torch.randn(1, 64)
