
class Model(torch.nn.Module):
    def __init__(self, num_heads, dropout_p):
        super().__init__()
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.softmax_qk = torch.nn.Softmax(dim=-1)
 
    def forward(self, query, key, value):
        scaled_qk = torch.matmul(query, key.transpose(-1, -2)) * (query.size(-1) ** -0.5)
        softmax_qk = self.softmax_qk(scaled_qk)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = torch.matmul(dropout_qk, value)
        return output

# Initializing the model
m = Model(num_heads=8, dropout_p=0.6)

# Inputs to the model
query = torch.randn(1, 8, 64, 64)
key = torch.randn(1, 8, 64, 64)
value = torch.randn(1, 8, 64, 64)
