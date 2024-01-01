
class Model(torch.nn.Module):
    def __init__(self, in_features, out_features, min_value=-.5, max_value=.5):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
        self.min_ = min_value
        self.max_ = max_value
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min_=self.min_)
        v3 = torch.clamp_max(v2, max_=self.max_)
        return v3

# Initializing the model
m = Model(in_features=6, out_features=8)

# Inputs to the model
x1 = torch.randn(1, 6)
