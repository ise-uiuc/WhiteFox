
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(p=dropout_p)
 
    def forward(self, q, k, v, scale_factor=1):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = (qk * scale_factor).softmax(dim=-1)
        dropout_qk = self.dropout(scaled_qk)
        output = self.dropout(scaled_qk).matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 10, 20)
k = torch.randn(1, 15, 25)
v = torch.randn(1, 30, 25)
