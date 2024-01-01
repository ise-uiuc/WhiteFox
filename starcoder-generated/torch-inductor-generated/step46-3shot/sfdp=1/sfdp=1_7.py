
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inv_scale_factor = 1 / math.sqrt(512)
        self.dropout_p = 0.3
        self.weights = torch.nn.Parameter(torch.Tensor())
 
    def forward(self, *inputs):
        q, k, v = inputs
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(3, 4, 512)
k = torch.randn(3, 4, 512)
v = torch.randn(3, 4, 512)
