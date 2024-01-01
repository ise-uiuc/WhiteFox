
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.nn.functional.linear(x1, torch.randn(4), torch.tensor([0.0, 1.0, 2.0, 3.0]))
        v1 = x1.permute(1, 2, 0)
        x2 = x2 + v1
        x2 = torch.nn.functional.relu(x2)
        return x2
# Inputs to the model
x1 = torch.randn(5, 2, 2)
