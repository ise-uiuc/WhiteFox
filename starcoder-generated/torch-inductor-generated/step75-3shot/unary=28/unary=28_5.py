
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(96, 96)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min_value=args.min_value)
        v3 = torch.clamp_max(v2, max_value=args.max_value)
        return v3

# Initialize the model
model = Model(**args.kw_args)

# Inputs to the model
x1 = torch.randn(1, 96)
