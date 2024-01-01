
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        x1 = torch.bmm(x1, x2)
        x1 = torch.nn.functional.relu(x1)
        return x1
# Inputs to the model
x1 = torch.randn(4, 2, 3)
x2 = torch.randn(4, 2, 5)
