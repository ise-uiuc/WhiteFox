
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = self.key = self.value = torch.nn.Parameter(torch.randn([8, 8]))
 
    def forward(self, x1):
        qk = torch.matmul(x1, self.key.transpose(-2,-1))
        scaled_qk = qk / math.sqrt(self.query.shape[-1])
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.3)
        output = dropout_qk.matmul(self.value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 8, 16)
