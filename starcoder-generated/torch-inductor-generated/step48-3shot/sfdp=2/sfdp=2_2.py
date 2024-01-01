
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        m1 = torch.matmul(x1, x2.transpose(-2, -1))
        m2 = m1 / 2.0
        m3 = torch.nn.functional.softmax(m2, dim=-1)
        m4 = torch.nn.functional.dropout(m3, p=0.5)
        v1 = torch.matmul(m4, x2)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 768, 500)
x2 = torch.randn(1, 768, 500)
