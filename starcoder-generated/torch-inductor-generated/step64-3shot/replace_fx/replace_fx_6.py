
class Res(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x: Tensor, x2: Tensor) -> Tensor:
        return x + torch.tanh(x2)
# Inputs to the model
x = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
x2 = torch.tensor([[1.5], [3.5]])
