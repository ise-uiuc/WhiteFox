
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.randn(4, 4)

    def forward(self, x1, x2):
        v1 = torch.matmul(x2, self.weight.transpose(0, 1))
        v2 = v1 * 0.125
        v3 = torch.nn.functional.softmax(v2, dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=0.7)
        v5 = torch.matmul(v4, x1)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4, 32)
x2 = torch.randn(1, 3, 4)
