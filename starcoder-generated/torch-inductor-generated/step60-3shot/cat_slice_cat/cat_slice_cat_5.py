
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        size = 10
        # Slice the input according your requirement
        x1 = input[:, 0:size]
        # Save the intermediate result
        x2 = input
        x3 = x2[:, 0:9223372036854775807]
        # Concatenate the results of the two slicing operations
        x4 = torch.cat([x1, x3], dim=1)
        return x4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 100)
