
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=-1)
 
    def forward(self, q, k, scale_factor, dropout_p):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = scale_factor * qk
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
model = Model()

# Inputs to the model
q = torch.randn(1, 6, 64)
k = torch.randn(1, 7, 64)
v = torch.randn(1, 7, 256)
scale_factor = 1.0
dropout_p = 0.0
