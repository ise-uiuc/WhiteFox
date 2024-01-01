
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
scale_factor = torch.zeros(256, 256)
__scaled_qk__ = torch.matmul(query, key.transpose(-2, -1)).mul(scale_factor)
dropout_p = 0.02
__dropout_qk__ = torch.nn.functional.dropout(__scaled_qk__.softmax(dim=-1), p=dropout_p)
