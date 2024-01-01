
class Model(torch.nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.dropout_p = 0.1
        self.head_dim = 32
        self.output_dim = self.num_heads * self.head_dim
 
    def forward(self, x1, x2, x3):
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        inv_scale_factor = 1. / np.sqrt(self.head_dim)
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(x3)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(12, 48, 32)
x2 = torch.randn(12, 32, 24)
x3 = torch.randn(12, 24, 64)
