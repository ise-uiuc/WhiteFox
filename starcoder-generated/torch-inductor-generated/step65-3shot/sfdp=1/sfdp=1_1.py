
class Model(torch.nn.Module):
    def __init__(self, d_k, dropout_p):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=1)
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, q, k, v, dropout_p):
        qk = torch.matmul(q, k.transpose(-2, -1))
        inv_scale_factor = float(d_k) ** -0.5
        scaled_qk = qk * inv_scale_factor
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model(d_k, dropout_p)
model = m.eval()

# Inputs to the model
query = torch.randn(3, 2, 768)
key = torch.randn(3, 13, 768)
value = torch.randn(3, 13, 768)
