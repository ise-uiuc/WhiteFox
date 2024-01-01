
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
 
    def forward(x1, x2, x3):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(0.1)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(768, 196)
x2 = torch.randn(768, 1)
x3 = torch.randn(1, 768)
