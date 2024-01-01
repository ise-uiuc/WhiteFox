
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1.div(0.25)
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=0.25)
        v5 = v4.matmul(x3)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 5, 3)
x2 = torch.randn(5, 4)
x3 = torch.randn(4, 6)
