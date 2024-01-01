
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(q_dropout_p)
        self.softmax = torch.nn.Softmax(dim=[-1])
    
    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scale_factor = 1.0 / math.sqrt(math.sqrt(float(k.shape[-1])))
        scaled_qk = qk * scale_factor
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = torch.matmul(dropout_qk, v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 64, 128)
k = torch.randn(1, 64, 128)
v = torch.randn(1, 64, 128)
