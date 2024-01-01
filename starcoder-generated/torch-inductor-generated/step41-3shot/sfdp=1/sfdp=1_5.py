
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = Param(torch.randn(4096, 512))
        self.key = Param(torch.randn(4096, 512))
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(dropout_p)
        self.value = Param(torch.randn(512, 4096))
 
    def forward(self, in1, in2):
        qk = torch.matmul(in1, in2.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = self.softmax(scaled_qk)
        output = self.dropout(softmax_qk).matmul(self.value)
        return output

# Initializing the model
m = Model()

# Input tensors to the model
x1 = torch.randn(2048, 512)
x2 = torch.randn(2048, 512)
