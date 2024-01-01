
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x1 = torch.matmul(x, torch.transpose(x, 0, 1))
        x1 = F.dropout(x1)
        y = torch.matmul(x, torch.transpose(x, 0, 1))
        return x1 + y
# Inputs to the model
x1 = torch.randn(3, 5)
