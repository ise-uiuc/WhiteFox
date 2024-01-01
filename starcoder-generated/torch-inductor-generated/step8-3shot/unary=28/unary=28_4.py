
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        # The linear transformation matrix is initialized to a
        # small matrix
        self.linear = torch.nn.Linear(64, 32, bias=False)
        self.linear.weight.data.fill_(-1.0)
        # The minimum and maximum values are stored in a private
        # variable (that is not learned)
        self.min_value, self.max_value = min_value, max_value
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3

# Values of the minimum and maximum
min_value, max_value = 0.01, 0.5
# Initializing the model
m = Model(min_value=min_value, max_value=max_value)

# Inputs to the model
x1 = torch.randn(1, 64)
