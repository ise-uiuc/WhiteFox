
class Model(torch.nn.Module):
    def __init__(self, d_model=2, dropout=0.):
        super().__init__()
        self.d_model = d_model
        self.dropout_p = dropout
 
  def forward(self, q, k, v, mask=None):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scale_factor = math.sqrt(self.d_model)
        inv_scale_factor = 1.0 / scale_factor
        scaled_qk = qk.div(inv_scale_factor)
        if self.dropout_p == 0:
            dropout_qk = scaled_qk.softmax(dim=-1)
        else:
            dropout_qk = torch.nn.functional.dropout(scaled_qk.softmax(dim=-1),
                p=self.dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 4, 2)
x3 = torch.randn(2, 3, 2)
x4 = torch.randn(2, 4, 2)
