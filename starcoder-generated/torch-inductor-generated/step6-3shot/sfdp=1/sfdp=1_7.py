
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(0.02)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.10)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()
 
# Inputs to the model
query = torch.randn(1, 3, 256, 256)
key = torch.randn(1, 3, 256, 256)
value = torch.randn(1, 3, 256, 256)
