
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor, other = torch.Tensor([1])) -> torch.Tensor:
        x = self.linear(input)
        return x + other

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 128, 2)
t = torch.randn(1, 128, 2)
