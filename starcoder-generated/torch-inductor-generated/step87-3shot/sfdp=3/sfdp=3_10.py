
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mat1 = torch.nn.Linear(8, 8, bias=False)
        self.mat2 = torch.nn.Linear(8, 8, bias=False)
        self.mat3 = torch.nn.Linear(8, 8, bias=False)
        self.mat4 = torch.nn.Linear(8, 8, bias=False)
        self.mat5 = torch.nn.Linear(8, 8, bias=False)
 
    def forward(self, x1, x2, x3, x4, x5, x6, x7, x8):
        q = self.mat1(x1)
        k = self.mat2(x2)
        v = self.mat3(x3)
        scale_factor = self.mat4(x4)
        dropout_p = self.mat5(x5)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scqkv = qk * scale_factor
        softmax_scqkv = scqkv.softmax(dim=-1)
        dropout_scqkv = torch.nn.functional.dropout(softmax_scqkv, p=dropout_p)
        mv = dropout_scqkv.matmul(v)
        return mv

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(10, 8)
x2 = torch.randn(10, 8)
x3 = torch.randn(10, 8)
x4 = torch.randn(10, 8)
x5 = torch.randn(10, 8)
x6 = torch.randn(1, 8)
x7 = torch.randn(1, 8)
x8 = torch.randn(1, 8)
