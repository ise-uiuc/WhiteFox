
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.nn.functional.dropout(x1, p=0.4)
        x3 = torch.nn.functional.dropout(x1, p=0.3)
        x4, x5, x6 = torch.chunk(x2, 3, 2)
        print(x1.shape, x2.shape, x3.shape, x4.shape, x5.shape, x6.shape)
        return x6.sum()
# Inputs to the model
x1 = torch.randn(1, 2, 2)
