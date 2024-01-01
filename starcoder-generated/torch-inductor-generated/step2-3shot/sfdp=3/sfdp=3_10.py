
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(x1, x2, x3, x4):
        v1 = torch.matmul(x2, x4.transpose(-2, -1))
        v2 = v1 * x3
        v3 = torch.nn.functional.softmax(v2, dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=x1)
        v5 = torch.matmul(v4, x4)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.rand([])
x2 = torch.randn(1, 20, 30) # x2's shape is [batch_size, src_seq_len, hidden_size]
x3 = torch.rand([])
x4 = torch.randn(1, 30, 20) # x4's shape is [batch_size, hidden_size, src_seq_len]
