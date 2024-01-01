
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.key = torch.nn.Parameter(torch.randn(8, 32, 32, 32))
 
    def forward(self, x1):
        query = x1
        key = self.key
        scale_factor = torch.tensor([10000.0])
        dropout_p = 0.1
        v1 = torch.matmul(query, key.transpose(-2, -1))
        v2 = v1 * scale_factor
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=dropout_p)
        v5 = torch.matmul(v4, value)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
batch_size = 1
seq_len = 10
feature = 16
x1 = torch.randn(batch_size, seq_len, feature)
