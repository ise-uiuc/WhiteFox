
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, dropout_p):
        scale_factor = torch.sqrt(query.size(-1))
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor) 
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

def generate():
    query = torch.randn(2, 3, 4, 5)
    key = torch.randn(2, 6, 4, 5)
    value = torch.randn(2, 6, 4, 5)
    model = Model()
    return __output__, query, key, value, __dropout_p__

# Initializing the model
__output__, __query__, __key__, __value__, __dropout_p__ = generate()
__output__, __query__, __key__, __value__, __dropout_p__ = generate()
__output__, __query__, __key__, __value__, __dropout_p__ = generate()

# Inputs to the model
__output__, __query__, __key__, __value__, __dropout_p__ = generate()

