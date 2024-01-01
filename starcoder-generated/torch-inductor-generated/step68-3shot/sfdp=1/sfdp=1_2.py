
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.3
 
    def forward(self, q1, k1, v1, scale_factor):
        qk = torch.matmul(q1, k1.transpose(-2, -1))
        scaled_qk = qk.div(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout.matmul(v1)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q1 = torch.randn(1, 16, 4)
k1 = torch.randn(1, 10, 4)
v1 = torch.randn(1, 10, 4)
scale_factor = 10
