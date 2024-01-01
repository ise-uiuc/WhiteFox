
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, qk, dropout):
        softmax_qk = torch.softmax(qk * 0.5, dim=-1)
        output = torch.nn.functional.dropout(softmax_qk, p=dropout).matmul(value)
        return output

# Initializing and setting parameters
m = Model()
dropout = 0.2
scale_factor = 0.2
query = torch.randn(1, 32, 128)
key = torch.randn(1, 32, 256)
value = torch.randn(1, 256, 64)
