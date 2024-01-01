
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x0):
        v1 = torch.nn.functional.linear(x0, torch.eye(2), bias=torch.zeros(2))
        v2 = x0.permute(0, 2, 1)
        return v1
# Inputs to the model
x0 = torch.randn(1, 2, 2)
