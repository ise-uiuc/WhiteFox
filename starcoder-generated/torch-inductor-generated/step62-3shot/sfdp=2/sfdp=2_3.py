
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, *args, **kwargs):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk / 1  # Set a scale factor that the softmax does not change the order of the tokens.
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.5)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 4, 2)
key = torch.randn(1, 8, 2) 
value = torch.randn(1, 8, 4)
state = {}
