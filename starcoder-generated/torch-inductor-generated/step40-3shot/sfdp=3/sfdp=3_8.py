
class Model(torch.nn.Module):
    def __init__(self, scale_factor, dropout_p):
        super().__init__()
        self.dropout_p = dropout_p
 
    def forward(self, q, k, v):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(scale_factor=14, dropout_p=0.8)

# Inputs to the model
query = torch.randn(4, 3, 64, 64)
key = torch.randn(4, 3, 64, 64)
value = torch.randn(4, 3, 64, 64)
