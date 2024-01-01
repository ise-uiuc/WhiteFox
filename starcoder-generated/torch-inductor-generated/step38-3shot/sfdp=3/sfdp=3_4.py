
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4, x5):
        v1 = torch.matmul(x1, x2.permute(0, 1, 3, 2))
        v2 = v1 * x3
        v3 = torch.nn.functional.softmax(v2, dim=-1)
        v4 = torch.nn.functional.dropout(v3, x4)
        v5 = v4.matmul(x5)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4, 8, 6)
x2 = torch.randn(1, 4, 6, 8)
x3 = torch.randn(1)
x4 = torch.randn(1)
x5 = torch.randn(1, 7, 2)
