
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
query = torch.randn(1, 256, 2048)
key = torch.randn(1, 256, 2048)
value = torch.randn(1, 256, 2048)
dropout_p = 0.125
scale_factor = torch.Tensor([5000.0])
