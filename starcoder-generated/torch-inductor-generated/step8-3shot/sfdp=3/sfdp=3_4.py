
class Model(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1 * math.sqrt(self.dim)
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=0.2, training=self.training)
        return v4.matmul(value)


# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(32, 20, dim)
x2 = torch.randn(32, dimension, self.hidden_size)
