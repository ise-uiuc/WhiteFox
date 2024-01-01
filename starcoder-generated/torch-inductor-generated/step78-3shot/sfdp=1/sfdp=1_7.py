
class Model(torch.nn.Module):
    def __init__(self, num_heads: int):
        self.num_heads = num_heads
        self.query = torch.nn.Linear(64, 64 * num_heads)
        self.key = torch.nn.Linear(64, 64 * num_heads)
        self.value = torch.nn.Linear(64, 64 * num_heads)
        self.scale_factor = 1.0 / (64 ** (1 / 4))
    
    def forward(self, query, key, value, dropout=0.):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(num_heads=4)

# Inputs to the model
query = torch.randn(1, 16, 64)
key = torch.randn(1, 16, 64)
value = torch.randn(1, 16, 64)
