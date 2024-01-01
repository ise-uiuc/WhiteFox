
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, q, k, v, dropout_p):
        output = torch.matmul(q, k.transpose(-2, -1))
        output = output * (int)(scale_factor)
        output = output.softmax(dim=-1)
        output = torch.nn.functional.dropout(output, p=dropout_p)
        output = torch.matmul(output, v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(4, 2, 64)
k = torch.randn(4, 64, 64)
v = torch.randn(4, 64, 64)
dropout_p = 0.2
