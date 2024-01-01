
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.randn(1, 8, 64, 64)
        self.key = torch.randn(1, 16, 32, 32)
        self.value = torch.randn(1, 16, 32, 32)
        self.scale_factor = 2 ** 3 # Scale factor used in the dot product
        self.inv_scale_factor = 1 / self.scale_factor # Inverse scale factor used in the dot product
        self.dropout_p = 0.5
 
    def forward(self, x1, x2, x3):
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(x3)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
x2 = torch.randn(1, 16, 32, 32)
x3 = torch.randn(1, 16, 32, 32)
