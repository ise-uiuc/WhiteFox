
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, scale_factor, dropout_p=0.1):
        qk = torch.matmul(query, key.transpose(-2, -1))
        qk = qk * scale_factor
        softmax_qk = torch.nn.functional.softmax(qk, dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        return torch.matmul(dropout_qk, value)

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(3, 5, 7)
key = torch.randn(3, 7, 11)
value = torch.randn(3, 5, 11)
