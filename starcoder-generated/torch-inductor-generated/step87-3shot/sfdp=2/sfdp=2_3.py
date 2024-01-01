
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Parameter(torch.randn(8, 8))
        self.key = torch.nn.Parameter(torch.randn(8, 8))
        self.value = torch.nn.Parameter(torch.randn(8, 8))
        self.inv_scale_factor = torch.nn.Parameter(torch.randn(()))
        self.dropout_p = torch.nn.Parameter(torch.randn(()))
 
    def forward(self, x1):
        qk = torch.matmul(self.query, self.key.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, self.dropout_p)
        result = dropout_qk.matmul(self.value)
        return result

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8, 128)
