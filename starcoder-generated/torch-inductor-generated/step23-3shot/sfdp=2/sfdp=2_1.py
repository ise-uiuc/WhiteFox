
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(-2, -1))
        sf = (1.0 / math.sqrt(dim[1]))
        scaled_qk = qk * sf
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
dim = [4, 5]
dropout_p = 0.5
q = torch.randn(1, dim[0])
k = torch.randn(1, dim[1], dim[2])
v = torch.randn(1, dim[2], dim[3])
