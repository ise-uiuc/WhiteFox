
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 1.0
        self.scale_factor = 1024
        self.dropout = torch.nn.Dropout(self.dropout_p)
      
    def forward(self, v1):
        q1 = torch.randn(1, 512, 64)
        k = v1
        v = v1
        q = q1
        inv_scale_factor = 1.0 / self.scale_factor
        qk = torch.matmul(q, k.transpose(1, 2))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 512, 64)
