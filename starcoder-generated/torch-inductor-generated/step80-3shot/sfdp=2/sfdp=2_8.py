
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(16)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model. 2 queries with size 16, 2 keys with size 16, and 2 values with size 16
query = torch.randn(2, 16, 2)
key = torch.randn(2, 2, 16)
value = torch.randn(2, 2, 16)
