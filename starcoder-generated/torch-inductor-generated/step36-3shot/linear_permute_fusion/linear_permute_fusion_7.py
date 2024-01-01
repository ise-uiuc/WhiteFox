
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x3):
        v1 = torch.nn.functional.softmax(x3, dim=2)
        return v1
# Inputs to the model
x3 = torch.rand(1, 2, 2, dtype=torch.float64)
