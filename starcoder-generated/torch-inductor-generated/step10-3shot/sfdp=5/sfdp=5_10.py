
class Model(torch.nn.Module):
    def forward(self, x1, x2):
        qk = x1 @ x2.transpose(-2, -1) / math.sqrt(x1.size(-1))
        qk = qk + x3 = torch.softmax(qk, dim=-1)
        qk = torch.dropout(qk, 0.1)
        output = qk @ x4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 4, x)
x2 = torch.randn(4, 4, y)
x3 = torch.randn(4, 4, z)
x4 = torch.randn(4, z, y)
