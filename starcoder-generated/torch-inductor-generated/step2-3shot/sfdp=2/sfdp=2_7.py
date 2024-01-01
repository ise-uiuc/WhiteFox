
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=0.4)

    def forward(self, x1, x2):
        x = torch.matmul(x1, x2.transpose(-2, -1))
        x = x / 32 / 32
        x = self.dropout(torch.nn.functional.softmax(x, dim=-1))
        x = torch.matmul(x, x2)
        return x

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 1, 32, 32)
x2 = torch.randn(2, 1, 32, 32)
