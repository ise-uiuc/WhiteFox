
class Model(torch.nn.Module):
    def forward(self, x, y):
        v1 = torch.matmul(x, y.transpose(-2, -1))
        v2 = v1 * 3
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=0.25)
        v5 = torch.matmul(v4, x)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 64, 2)
y = torch.randn(1, 2, 4)
