
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, scale_factor=1.0, dropout_p=0.0):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
torch.manual_seed(0)
m = Model()

# Inputs to the model
query = torch.randn(1, 32, 64)
key = torch.randn(1, 32, 64)
value = torch.randn(1, 32, 64)
