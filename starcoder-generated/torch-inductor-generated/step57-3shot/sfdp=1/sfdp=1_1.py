
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.2
        self.dropout = torch.nn.Dropout(self.dropout_p)

    def forward(self, q, k, v, isf=8):
        x1 = torch.matmul(
            query,
            key.transpose(-2, -1),
        )
        x2 = x1.div(isf)
        x3 = torch.nn.Softmax(x2)
        x4 = self.dropout(x3)
        x5 = torch.matmul(
            x4,
            value,
        )
        return x5

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 23, 128, 128)
key = torch.randn(1, 23, 128, 128)
value = torch.randn(1, 23, 128, 128)
