
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Parameter(torch.randn(1, 64, 32, 32))
        self.key = torch.nn.Parameter(torch.randn(1, 32, 64, 64))
        self.value = torch.nn.Parameter(torch.randn(1, 32, 64, 64))
        self.inv_scale_factor = torch.nn.Parameter(torch.randn(32, 32))
        self.dropout_p = 0.8
 
    def forward(self, x2):
        qk = torch.matmul(self.query, self.key.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(self.value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 1, 64, 64)
