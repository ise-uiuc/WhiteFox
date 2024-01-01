
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=-1)
 
    def forward(self, x1, x2):
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        scaling_factor = torch.tensor([10.0, 20.0], dtype=torch.float32)
        scaled_qk = qk * scaling_factor.view(1, -1, 1)
        softmax_qk = self.softmax(scaled_qk)
        dropout_p = 0.2
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        out = dropout_qk.matmul(x2)
        return out

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10, 64, 64)
x2 = torch.randn(1, 5, 64, 64)
