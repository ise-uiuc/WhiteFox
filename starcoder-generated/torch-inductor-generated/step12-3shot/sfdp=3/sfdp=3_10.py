
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.scale_factor = torch.tensor(1.0 / math.sqrt(768))
        # self.scale_factor = torch.scalar_tensor(1.0 / math.sqrt(768))
     
    def forward(self, x, y):
        qk = torch.matmul(x, y.T)
        scaled_qk = qk * self.scale_factor
        softmax_qk = softmax(scaled_qk.transpose(-2, -1)).transpose(-2, -1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.5, training=self.training)
        output = torch.matmul(dropout_qk, y)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(64, 768)
y = torch.randn(768, 2048)
