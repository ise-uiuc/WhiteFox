
class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x, y, z):
        qk = torch.matmul(x, y.transpose(-2, -1))
        scaled_qk = qk.div(1/math.sqrt(2*math.pi))
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.1)
        w = dropout_qk[0]
        output = torch.matmul(w, z)
        return output

# Initializing the model
m = Model2()

# Inputs to the model
x = torch.randn(1, 8, 4, 4)
y = torch.randn(1, 8, 4, 4)
z = torch.randn(1, 8, 4, 4)
output = m(x, y, z)

