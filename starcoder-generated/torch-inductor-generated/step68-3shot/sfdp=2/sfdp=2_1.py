
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.5)
 
    def forward(self, q, k, v, isf):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(isf)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 3, 64, 64)
k = torch.randn(1, 3, 21, 21)
v = torch.randn(1, 3, 21, 21)
isf = torch.tensor([1.0])
