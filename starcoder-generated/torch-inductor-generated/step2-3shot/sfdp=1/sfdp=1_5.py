
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x10, scale_factor, dropout_p):
        v1 = torch.matmul(x1, x10.transpose(-2, -1))
        v2 = v1 / scale_factor
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=dropout_p)
        v5 = v4.matmul(x1)
        return v5

# Initializing the model
m = Model()

# Data to the model
x1 = torch.randn(2, 4, 3)
x10 = torch.randn(3, 4, 7)

scale_factor = torch.randn(1).abs()
dropout_p = torch.randn(1).abs()

# Run the model
