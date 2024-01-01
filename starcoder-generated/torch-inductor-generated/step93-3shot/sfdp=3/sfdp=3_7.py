
class Model(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.scale_factor = torch.tensor([1.0 / math.sqrt(self.config.d_model)])
        self.dropout_p = config.dropout_p
 
    def forward(x1, x2, x3):
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(x3)
        return output

# Initializing the model
config = Config()
m = Model(config)

# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
x2 = torch.randn(1, 8, 64, 64)
x3 = torch.randn(1, 8, 64, 64)
