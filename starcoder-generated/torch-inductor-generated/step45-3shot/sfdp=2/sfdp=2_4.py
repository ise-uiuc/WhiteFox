
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(0.1)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.5)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 3, 16, 4)
key = torch.randn(1, 4, 128, 512)
value = torch.randn(1, 3, 16, 512)
