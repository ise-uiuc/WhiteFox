
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mat1 = torch.arange(64, dtype=torch.float).reshape(1, 64)
        self.mat2 = torch.arange(64, dtype=torch.float).reshape(64, 1)
        self.scale_factor = 128
        self.dropout_p = 0.5
 
    def forward(self, x1):
        qk = torch.matmul(x1, self.mat2)
        scaled_qk = qk.div(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, self.dropout_p)
        output = dropout_qk.matmul(self.mat1)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
