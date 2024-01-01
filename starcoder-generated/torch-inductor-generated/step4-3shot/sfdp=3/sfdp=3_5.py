
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout2d(p=0.0)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.matmul1 = torch.nn.MatMul()
        self.matmul2 = torch.nn.MatMul()
        self.mul = torch.mul
        self.add = torch.add
 
    def forward(self, q, k, v, scale_factor):
        qk = self.matmul1(q, k)
        scaled_qk = self.mul(qk, scale_factor)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = self.matmul2(dropout_qk, v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(10, 5)
k = torch.randn(10, 5)
v = torch.randn(10, 5)
scale_factor = np.asarray(np.random.uniform(low=1, high=100, size=1))

