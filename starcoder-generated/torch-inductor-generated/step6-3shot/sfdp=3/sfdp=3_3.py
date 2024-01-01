
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, query, key, value, scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
dropout_p, scale_factor = 0.1, 0.1
m = Model()

# Inputs to the model
query, key, value = torch.randn(3, 1, 8, 8), torch.randn(3, 1, 8, 8), torch.randn(3, 1, 8, 8)
