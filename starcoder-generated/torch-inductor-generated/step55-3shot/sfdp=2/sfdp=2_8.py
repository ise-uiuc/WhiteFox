
class Model(torch.nn.Module):
    def __init__(self)
        inv_scale_factor = 1.0 / math.sqrt(512)
        dropout_p = 0.1
        super().__init__()
        self.dot_product = torch.nn.DotProduct() 
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, x1, x2):
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = torch.matmul(dropout_qk, x2)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 512, 16)
x2 = torch.randn(1, 512, 9)
