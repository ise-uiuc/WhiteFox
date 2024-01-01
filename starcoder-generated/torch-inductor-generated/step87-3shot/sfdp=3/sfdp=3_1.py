
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(query, key, value, dropout_p, scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk * scale_factor
        softmax_qk = torch.nn.functional.softmax(scaled_qk, dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk * value
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 5, 100)
key = torch.randn(1, 5, 100)
value = torch.randn(1, 5, 100)
dropout_p = 0.3
__scale_factor__ = torch.sqrt(query.size(-1))
