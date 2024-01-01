
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Parameter(torch.randn(51, 8, 2, 2))
        self.scale_factor = torch.nn.Parameter(torch.FloatTensor([math.sqrt(1.0 / 2.0)]))
        self.value = torch.nn.Parameter(torch.randn(100, 51, 4, 4))
 
    def forward(self, x2):
        q = self.query
        scale_factor = self.scale_factor
        v = self.value
        qk = torch.matmul(q, q.transpose(-2, -1))
        scaled_qk = qk.div(scale_factor)
        dropout_qk = torch.nn.functional.dropout(scaled_qk, p=0.5)
        output = dropout_qk.matmul(v)
        return output
        
# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 51, 2, 2)
m(x2)

