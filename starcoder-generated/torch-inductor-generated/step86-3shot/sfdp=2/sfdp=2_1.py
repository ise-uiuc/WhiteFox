
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(p=0.5)
 
    def forward(self, q1, k1):
        qk = torch.matmul(q1, k1.transpose(-2, -1))
        inv_scale_factor = 1 / qk.size(-1) ** 0.25
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(v1)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q1 = torch.randn(128, 12, 768)
k1 = torch.randn(128, 12, 768)
v1 = torch.randn(128, 12, 768)
