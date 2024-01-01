
class Model(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.matmul1 = torch.nn.Linear(config.d, config.d)
        self.matmul2 = torch.nn.Linear(config.d, config.d)
        self.matmul3 = torch.nn.Linear(config.d, config.d)
        self.matmul4 = torch.nn.Linear(config.d, config.d)
        self.matmul5 = torch.nn.Linear(config.d, config.d)
 
    def forward(self, x, y):
        q, k, v = self.matmul1(x), self.matmul2(y), self.matmul3(y)
        qy = self.matmul4(x)
        inv_scale_factor = self.matmul5(x)
        qk = torch.matmul(qy, k.transpose(-2, -1))
        qk = qk.div(inv_scale_factor)
        softmax_qk = qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=config.attention_dropout_prob)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
conf = Config()
m = Model(conf)

# Inputs to the model
x, y = torch.randn(conf.batch_size, conf.d), torch.randn(conf.batch_size, conf.d)
