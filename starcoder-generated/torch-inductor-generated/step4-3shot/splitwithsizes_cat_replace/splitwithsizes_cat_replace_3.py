
class Model(torch.nn.Module):
    def forward(self, x1):
        split_tensors = torch.split(x1, [1, 1, 1], dim=3)
        return split_tensors

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1, 1, 3)
