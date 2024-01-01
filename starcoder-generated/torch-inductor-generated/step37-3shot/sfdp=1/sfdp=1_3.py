
class Model(torch.nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.w = torch.randn(3, 3, 1, 1)
 
    def forward(self, x1):
        qk = torch.matmul(x1, self.w.transpose(-2, -1))
        scaled_qk = qk.div(self.config.div)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.config.dropout)
        output = dropout_qk.matmul(x1)
        return output

# Initializing the model
m = Model(config=Configs()).to(device)

# Inputs to the model
x1 = torch.randn(1, 3)
