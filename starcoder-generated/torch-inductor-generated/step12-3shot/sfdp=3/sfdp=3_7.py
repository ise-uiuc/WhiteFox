
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, q, k, v):
        scale_factor = k.size(-1) ** 0.25
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk * scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 3, 2048)
k = torch.randn(1, 3, 512)
v = torch.randn(1, 3, 512)
