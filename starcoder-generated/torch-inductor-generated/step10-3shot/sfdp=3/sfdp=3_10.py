
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.1)
 
    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(-2, -1))
        dropout_qk = self.dropout(qk)
        output = torch.matmul(dropout_qk, v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 8, 96, 96)
k = torch.randn(1, 8, 96, 96)
v = torch.randn(1, 8, 96, 96)
