
class Model(torch.nn.Module):
    def __init__(self, n_heads=1, head_size=20, dropout_p=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_size // n_heads
        self.dropout_p = dropout_p

    def forward(self, query, key, value, attn_mask):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk / math.sqrt(self.head_dim)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 20, 5, 20)
key = torch.randn(1, 20, 10, 20)
value = torch.randn(1, 20, 10, 20)
attn_mask = torch.randn(1, 5, 10)
