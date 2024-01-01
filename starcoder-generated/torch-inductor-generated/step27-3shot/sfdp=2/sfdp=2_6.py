
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, q1, k1, v1):
        q = q1
        k = k1.transpose(-2, -1)
        v = v1
        scaled_qk = torch.matmul(q, k) / math.sqrt(q1.size(-1))
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.5)
        output = torch.matmul(dropout_qk, v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q1 = torch.randn(1, 16, 512)
k1 = torch.randn(1, 16, 512)
v1 = torch.randn(1, 16, 512)
