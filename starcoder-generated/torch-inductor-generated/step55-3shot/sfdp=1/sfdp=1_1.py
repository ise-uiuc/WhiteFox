
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_linear = torch.nn.Linear(10, 4)
        self.k_linear = torch.nn.Linear(10, 4)
        self.dropout = torch.nn.Dropout(.2)
 
    def forward(self, input1, input2):
        x1 = self.q_linear(input1)
        x2 = self.k_linear(input2)
        kq = torch.matmul(x1, x2.transpose(-2, -1))
        inv_scale_factor = 1. / math.sqrt(self.k_linear.in_features)
        scaled_qk = kq.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        v1 = torch.matmul(dropout_qk, x2)
        return v1

# Initializing the model and inputs
m = Model()
input1 = torch.randn(3, 4, 10)
input2 = torch.randn(3, 5, 10)
