
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, q, k, v, scale_factor, dropout_p):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output


# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 8, 12, 64)
key = torch.randn(1, 8, 64, 12)
value = torch.randn(1, 8, 12, 64)
scale_factor = 10.0
dropout_p = 0.01
