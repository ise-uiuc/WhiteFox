
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = softmax(scaled_qk, dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Input to the model
query = torch.randn(8, 12, 768)
key = torch.randn(8, 12, 768)
value = torch.randn(8, 12, 768)
scale_factor = torch.randn(8)
dropout_p = torch.nn.Parameter(torch.full((1,), 0.5, dtype=torch.float32))
