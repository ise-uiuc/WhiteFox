
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        inv_scale_factor = 2.0**0.5
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.5)
        output = dropout_qk.matmul(value)
        return output

# Initialize the model
m = Model()

# Inputs to the model
query = torch.randn(1, 8, 4, 4)
key = torch.randn(1, 8, 4, 4)
value = torch.randn(1, 8, 4, 4)
