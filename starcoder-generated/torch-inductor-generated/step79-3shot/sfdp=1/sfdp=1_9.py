
class Model(torch.nn.Module):
    def __init__(self, q, k, v, d):
        super().__init__()
        self.qkv_matmul = torch.matmul(q, k.transpose(-2, -1))
        self.qkv_div = self.qkv_matmul.div(d)
        self.softmax = torch.nn.Softmax()
        self.dropout = torch.nn.Dropout(0.8)
        self.output_matmul = self.dropout(self.softmax(self.qkv_div))
        self.output = self.output_matmul.matmul(v)
 
    def forward(self, q, k, v):
        self.qkv_matmul = torch.matmul(q, k.transpose(-2, -1))
        self.qkv_div = self.qkv_matmul.div(d)
        self.softmax = torch.nn.Softmax()
        self.dropout = torch.nn.Dropout(0.8)
        self.output_matmul = self.dropout(self.softmax(self.qkv_div))
        self.output = self.output_matmul.matmul(v)
        return self.output

# Initializing the model
q = torch.tensor([[2, 4, 5]], dtype=torch.float32)
k = torch.tensor([[1.2, -3.4, 0.0, 5.2, 2.1], [0.1, 4.1, -2.3, 2.4, -3.8], [1, -100, -100, 1, -100]], dtype=torch.float32)
v = torch.tensor([[2, 0, 5, 1.2, -2.4], [1, 3.4, -0.1, 4.2, -2.2], [10, 20, 30, 40, 50]], dtype=torch.float32)
d = torch.tensor([30], dtype=torch.float32)
m = Model(q, k, v, d)

# Inputs to the model
x1 = torch.randn(3, 5)
