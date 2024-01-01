
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2, x3):
        v1 = x1.permute(0, 2, 1)
        v1 = torch.matmul(v1, x2)
        v1 = x1.permute(0, 2, 1)
        v1 = torch.matmul(v1, x3)
        v3 = x1.permute(0, 2, 1)
        v4 = x2.permute(0, 2, 1)
        v5 = x3.permute(0, 2, 1)
        v2 = torch.matmul(v3, v4)
        v2 = torch.matmul(v2, v5)
        return v1, v2
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 1, 2)
x3 = torch.randn(1, 2, 2)
out1, out2 = m(x1, x2, x3)

# ReLU6 with fusion pattern
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(3, 3, 1, 1, 0, 1, 1)

        self.relu6 = torch.nn.ReLU6(True)

    def forward(self, x):
        v1 = self.conv2d(x)
        v2 = self.relu6(v1)
        return v2
        
m = Model()

# Inputs to the model
x = torch.randn(2, 3, 2, 2)
out_cpu = m(x)

# PyTorch code after symbolic tracing
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(3, 3, 1, 1, 0, 1, 1)

        self.relu6 = torch.nn.ReLU6(True)

    def forward(self, x):
        v1 = self.conv2d(x)
        v2 = self.conv2d(v1)
        v2 = v2.clamp_(0, 6)
        return v2

# Initializing the model
m = Model()
input_x = torch.randn(2, 3, 2, 2, requires_grad = True)

# Inputs to the model
with torch.no_grad():
    out_cpu = m(input_x)

# PyTorch code after symbolic tracing
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(3, 3, 1, 1, 0, 1, 1)

        self.relu6 = torch.nn.ReLU6(True)

    def forward(self, x):
        v1 = self.conv2d(x)
        v2 = self.conv2d(v1)
        v2 = v2.clamp_(0, 6)
        return v2

# Initializing the model
m = Model()
input_x = torch.randn(2, 3, 2, 2, requires_grad = True)

# Inputs to the model
graph = torch.onnx.utils._model_to_graph(m, (input_x, ), False, False, 'trace_mode')

# Pytorch code after fusion
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(3, 3, 1, 1, 0, 1, 1)

        self.relu6 = torch.nn.ReLU6(True)

    def forward(self, x):
        v1 = self.conv2d(x)
        v2 = v1.clamp_(0, 6)
        return v2
        
# Initializing the model
m = Model()
input_x = torch.randn(2, 3, 2, 2, requires_grad = True)

m = torch.jit.script(m)

graph = torch.onnx.utils._model_to_graph(m, (input_x, ), False, False, 'eval')
