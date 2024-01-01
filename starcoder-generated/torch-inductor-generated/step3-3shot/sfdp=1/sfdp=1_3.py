
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inv_scale_factor = torch.tensor(64)
        self.dropout_p = torch.tensor(0.1)
  
    def forward(self, x_qk, x_v):
        qk = torch.matmul(x_qk, x_qk.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(x_v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x_qk = torch.randn(1, 4, 100)
x_v = torch.randn(1, 4, 256)
