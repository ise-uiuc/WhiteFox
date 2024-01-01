
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.rand([8, 64, 1, 128])
        self.key = torch.rand([8, 128, 1, 128])
        self.value = torch.rand([8, 128, 1, 128])
        self.dropout_p = 0.5
 
    def forward(self):
        qk = torch.matmul(self.query, self.key.transpose(-2, -1))
        inv_scale_factor = math.sqrt(128/128)
        v1 = qk.div(inv_scale_factor)
        softmax_qk = v1.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk.div(self.dropout_p), p=self.dropout_p)
        v2 = dropout_qk.matmul(self.value)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
