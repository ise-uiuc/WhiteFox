
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1 * 1/sqrt(800)
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=0.2)
        v5 = torch.matmul(v4, x3)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 400, 800)
x2 = torch.randn(1, 800, 400)
x3 = torch.randn(1, 400, 800)
