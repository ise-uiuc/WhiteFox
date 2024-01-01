
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.5)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.matmul1 = torch.matmul
 
    def forward(self, query, key, value, scale_factor, dropout_p):
        qk = self.matmul1(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = torch.matmul(dropout_qk, value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 4, 16)
key = torch.randn(1, 4, 128)
value = torch.randn(1, 4, 128)
scale_factor = 10
