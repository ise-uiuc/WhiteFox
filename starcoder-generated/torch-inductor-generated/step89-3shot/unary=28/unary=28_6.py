
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(size_features, num_output)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min_value=lower_bound)
        v3 = torch.clamp_max(v2, max_value=upper_bound)
        return v3

# Initializing the model
m = Model()
num_output = 10
size_features = 10
# Inputs to the model
x1 = torch.randn(2, size_features)
