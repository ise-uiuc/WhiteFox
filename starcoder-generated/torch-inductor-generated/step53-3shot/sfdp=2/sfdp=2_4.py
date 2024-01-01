
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query  = [1.0,1.0,1.0]
        self.key    = [[1.0,1.0,1.0],
                       [1.0,1.0,1.0]]
        self.value  = [[1.0,1.0,1.0],
                       [1.0,1.0,1.0]]

    def forward(self, qk, dropout_p):
        inv_scale_factor = 1.0 / math.sqrt(3)
        qk = torch.matmul(qk, self.key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(self.value)
        return output

# Initializing the model
m = Model()
# Inputs to the model'
qk = torch.randn(2, 3)
dropout_p = 0
