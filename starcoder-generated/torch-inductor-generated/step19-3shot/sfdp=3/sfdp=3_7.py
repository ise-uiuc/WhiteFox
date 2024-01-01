
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        q = x1
        k = x2
        scale_factor = 1 / math.sqrt(k.size(-1))
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk * scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.1)
        output = dropout_qk * v
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16, 512)
x2 = torch.randn(1, 16, 512)
