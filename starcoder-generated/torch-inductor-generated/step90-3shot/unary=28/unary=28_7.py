
def Model(min_value=-0.2, max_value=3.0):
    return torch.nn.Sequential(
        torch.nn.Linear(32, 64),
        MinMaxClamp_MinMax(min_value=min_value, max_value=max_value)
    )

# Initializing the model
m = Model(min_value=-1.0, max_value=3.0)

# Inputs to the model
x1 = torch.randn(32)
