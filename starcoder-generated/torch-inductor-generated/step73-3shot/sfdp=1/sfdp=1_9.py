
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.6983342451369904)
 
    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(1e-12)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(v)
        return output

# Initialize the model
m = Model()

# Inputs to the model
q = torch.randn(1, 8, 512)
k = torch.randn(1, 8, 512)
v = torch.randn(1, 8, 512)
