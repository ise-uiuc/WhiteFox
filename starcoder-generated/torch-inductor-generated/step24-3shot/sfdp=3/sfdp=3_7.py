
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=0.1)
        
    def forward(self, q, k, v):
        qk = q.matmul(k.transpose(-2, -1))
        scale_factor = 1.0 / math.sqrt(q.shape[-1])
        scaled_qk = qk * scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)      
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 4, 128)
k = torch.randn(1, 4, 128)
v = torch.randn(1, 4, 128)
