
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, matrix):
        v1 = torch.mm(matrix, torch.tensor(2.0))
        return v1
# Inputs to the model
matrix = torch.randn(5, 5)
inp1 = torch.randn(1, 1)
inp2 = torch.randn(1, 1)
