
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = 0.3

    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2.transpose(1, 2))
        v2 = v1 * 1.153
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=self.dropout, training=True)
        v5 = torch.matmul(v4, x2)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(8, 192, 256)
key = torch.randn(8, 256, 84)
