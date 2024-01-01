
# This is a different version of model but the permutation pattern also exists there
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v2 = x2.permute(0, 2, 3, 1)
        v1 = torch.matmul(v2, x1)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 4, 2, 2)
