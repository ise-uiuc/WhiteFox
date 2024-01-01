
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1d = torch.nn.Linear(768, 768)
 
    def forward(self, x1, x2):
        tmp1 = self.linear1d(x1)
        qk = torch.matmul(tmp1, x2.transpose(-2, -1))
        inv_scale_factor = math.sqrt(x2.shape[-1])
        scaled_qk = qk.div(inv_scale_factor)
        output = softmax_qk.matmul(x2).to(torch.float16)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(128, 768)
x2 = torch.randn(128, 128, 64)
