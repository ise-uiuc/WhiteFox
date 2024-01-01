
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, q, k):
        scaled_qk = torch.matmul(q, k.transpose(-2, -1)) * math.sqrt(float(q.size(-1)))
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.1)
        return dropout_qk.transpose(-2, -1).matmul(v)

# Initializing the model
m = Model()
 
# Inputs to the model
q = torch.randn(2, 8, 32, 32)
k = torch.randn(2, 7, 32, 32)
v = torch.randn(2, 7, 32, 32)
