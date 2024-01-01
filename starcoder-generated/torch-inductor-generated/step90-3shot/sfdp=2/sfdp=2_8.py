
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.drop = torch.nn.Dropout(0.1)
 
    def forward(self, qv, kv):
        qk = torch.matmul(qv, kv.transpose(-2, -1))
        scale_factor = 1 / math.sqrt(math.sqrt(qk.shape[-1])) / math.sqrt(math.sqrt(kv.shape[-1]))
        scaled_qk = qk * scale_factor
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.drop(softmax_qk)
        output = dropout_qk.matmul(kv)
        return output

# Initializing the model
m = Model()

# Inputs to the model
qv = torch.randn(1, 4, 5)
kv = torch.randn(1, 3, 5)
