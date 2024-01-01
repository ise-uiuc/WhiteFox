
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = dropout3(x1, p=0.5)
        return torch.nn.functional.one_hot(x2)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
