
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input: torch.Tensor) -> List[torch.Tensor]:
        return torch.split(input, [500000, 150000], 1)
# Inputs to the model
x1 = torch.randn(1, 5, 1)
