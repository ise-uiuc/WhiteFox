
class Model(torch.nn.Module):
    def __init__(self, min_value=0.8929750293254677):
        super().__init__()
        self.linear = torch.nn.Linear(3, 10)
        self.min_value = torch.Tensor([min_value])[0]
 
    def forward(self, x1):
        return torch.clamp_max(torch.clamp_min(self.linear(x1), self.min_value), 0.1005024975024975)

# Initializing the model
m = Model(min_value=0.8929750293254677)

# Inputs to the model
x1 = torch.randn(20, 3)
