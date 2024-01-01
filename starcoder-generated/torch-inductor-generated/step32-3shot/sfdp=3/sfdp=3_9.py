
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
m = Model()

# Inputs to the model
query = torch.randn(6, 3, 5)
__key__ = torch.randn(6, 4, 5)
__value__ = torch.randn(6, 4, 5)
__scale_factor__ = torch.randn(6, 3, 4)
__dropout_p__ = 0.2
