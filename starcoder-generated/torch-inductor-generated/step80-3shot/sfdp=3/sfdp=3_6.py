
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.2
        self.softmax_dim = -1
        self.scale_factor = 1 / math.sqrt(128)
        self.dropout = torch.nn.Dropout(self.dropout_p)
 
    def forward(self, x1, x2, x3):
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        scaled_qk = qk * self.scale_factor
        softmax_qk = torch.nn.functional.softmax(scaled_qk, dim=self.softmax_dim)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(x3)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 128, 512)
x2 = torch.randn(1, 128, 512)
x3 = torch.randn(1, 512, 512)
