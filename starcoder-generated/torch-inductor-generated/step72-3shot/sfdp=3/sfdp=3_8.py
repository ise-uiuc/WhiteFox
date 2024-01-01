
class Model(torch.nn.Module):
    def __init__(self, dropout_p):
        super().__init__()
        self.dropout_p = dropout_p
 
    def forward(self, query, key, value, scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk * scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        return dropout_qk.matmul(value)

# Initializing the model
m = Model(dropout_p)

# Inputs to the model
query = torch.randn(1, 16, 10)
key = torch.randn(1, 16, 20)
value = torch.randn(1, 16, 20)
scale_factor = torch.ones([1, 1, 1])
