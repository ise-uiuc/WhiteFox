
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=0.10)
    
    def query(self, input_tensor):
        return torch.randn(4, 8, 20)
    
    def key(self, input_tensor):
        return torch.randn(4, 4, 20)
    
    def value(self, input_tensor):
        return torch.randn(4, 4, 10)
    
    def forward(self, x1):
        q = self.query(x1)
        k = self.key(x1)
        v = self.value(x1)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(5.0)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.10)
        output = dropout_qk.matmul(v)
        return output.view(x1.shape)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 8, 10)
