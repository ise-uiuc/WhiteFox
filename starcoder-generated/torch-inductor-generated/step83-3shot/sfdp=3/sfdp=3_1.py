
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = dropout_p
        self.attention = MultiHeadSelfAttention(8, 32, dropout_p)
 
    def forward(self, query, key, value, mask):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scale_factor = 1 / math.sqrt(qk.shape[-1])
        qk = qk * scale_factor
        softmax_qk = qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        return dropout_qk.matmul(value)

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 8, 64, 64)
key = torch.randn(1, 8, 128, 64)
value = torch.randn(1, 8, 128, 64)
mask = torch.ones(1, 1, 64, 64)
