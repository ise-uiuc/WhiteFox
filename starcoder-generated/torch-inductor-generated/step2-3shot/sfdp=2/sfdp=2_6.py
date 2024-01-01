
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.k1 = torch.nn.Linear(8, 16, bias=False)
        self.q2 = torch.nn.Linear(8, 16, bias=False)
        self.v3 = torch.nn.Linear(8, 16, bias=False)
 
    def forward(self, k2, q3, v4):
        scaled_qk = torch.matmul(q3, self.k1(k2).transpose(-2, -1)).div(8)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, 0.4000000059604645)
        output = dropout_qk.matmul(self.v3(v4))
        return output

# Initializing the model
m = Model()

# Inputs to the model
k2 = torch.randn(1, 8, 128)
q3 = torch.randn(1, 8, 128)
v4 = torch.randn(1, 8, 128)
