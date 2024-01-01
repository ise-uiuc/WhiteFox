
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, alpha=1):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk * alpha
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.1)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(5, 8, 64)
key = torch.randn(5, 6, 64)
value = torch.randn(5, 6, 64)
