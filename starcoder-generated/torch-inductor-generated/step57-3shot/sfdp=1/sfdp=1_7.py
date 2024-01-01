
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(dropout_p)
  
    def forward(self, q, k, v, scale_factor):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(scale_factor)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = torch.matmul(dropout_qk, v)
        return output


# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 6, 128)
k = torch.randn(1, 20, 128)
v = torch.randn(1, 20, 128)
scale_factor = 128 ** -0.5
