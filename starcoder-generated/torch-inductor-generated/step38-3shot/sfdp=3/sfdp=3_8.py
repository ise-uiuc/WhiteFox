
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout_p)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.matmul = torch.matmul
 
    def forward(self, q, k, v, scale_factor):
        qk = self.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = self.matmul(dropout_qk, v)
        return output

# Initializing the model
model = Model()

# Inputs to the model
q = torch.tensor([[1, 2, 5]], dtype=torch.float)
k = torch.tensor([[3, 10, 7], [1, 1, 3], [0, -2, 4]], dtype=torch.float)
v = torch.tensor([[1, 8, -1], [0, 1, -3], [1, 1, -1]], dtype=torch.float)
scale_factor = 0.3
