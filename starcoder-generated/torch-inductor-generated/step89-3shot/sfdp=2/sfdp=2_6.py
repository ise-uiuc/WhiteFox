
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1 / 15
        v3 = SoftMax(dim=-1)(v2)
        v4 = torch.nn.functional.dropout(v3, p=0.6)
        return v4.matmul(x1)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(5, 4, 112, 112)
x2 = torch.randn(5, 4, 224, 224)
