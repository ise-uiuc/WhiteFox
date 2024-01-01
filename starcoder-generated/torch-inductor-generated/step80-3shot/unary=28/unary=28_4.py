
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(64, 128)
        self.minmax = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Hardtanh(min_value=min_value, max_value=max_value),
            # Note: If torch.nn.Hardtanh has only min_value or only max_value,
            # torch.nn.ReLU is not necessary.
        )
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = self.minmax(v1)
        return v2

# Initializing the model
m = Model(min_value=0., max_value=1.)

# Inputs to the model
x1 = torch.randn(1, 64)
