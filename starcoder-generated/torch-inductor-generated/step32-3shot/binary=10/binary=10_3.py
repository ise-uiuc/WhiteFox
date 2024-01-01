
def forward(input_tensor, other):
    v1 = self.___forward___(input_tensor)
    v2 = v1 + other
    return v2

# Initializing the model
linear = Linear(3, 128)

# Inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)
other = torch.randn(1, 128)
