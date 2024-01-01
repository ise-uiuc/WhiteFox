
class Model(torch.nn.Module):
    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1 * 10000.
        v3 = nn.functional.softmax(v2, -1)
        v4 = nn.functional.dropout(v3, 0.2)
        v5 = v4.matmul(x2)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8, 128, 128)
x2 = torch.randn(1, 512, 16, 16)
