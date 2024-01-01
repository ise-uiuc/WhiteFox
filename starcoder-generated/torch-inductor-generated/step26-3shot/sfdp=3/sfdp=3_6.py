
class Model(torch.nn.Module):
    def __init__(self):
        pass
 
    def forward(self, query, key, value, dropout_p, scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(3, 5, 7)
key = torch.randn(3, 5, 7)
value = torch.randn(3, 5, 7)
dropout_p = 0.8
scale_factor = 0.1
