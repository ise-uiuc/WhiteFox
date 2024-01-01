
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.softmax = torch.nn.Softmax()
    def forward(self, x1):
        v4 = x1
        v1 = torch.nn.functional.linear(v4, self.linear.weight, self.linear.bias)
        a1 = torch.tensor([[1, 2], [3, 4]], device='cpu', requires_grad=True)
        v2 = v1.permute(0, 2, 1)
        v3 = self.softmax(a1)
        return (v1, v2)
# Inputs to the model
x1 = torch.randn(1, 2, 2)

# This one has shape mismatch on the model inputs:
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 4)

# This one has a shape mismatch on model outputs:
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 4)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 2)

# This one has a shape mismatch on model outputs:
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 4)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        return (x1, v2)
# Inputs to the model
x1 = torch.randn(1, 2, 2)

# This one has a shape mismatch on model inputs and outputs:
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 4)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        return (x1, v1, v2)
# Inputs to the model
x1 = torch.randn(1, 2, 2)

# This one has a shape mismatch on model inputs and outputs:
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 4)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        return (v2, v1)
# Inputs to the model
x1 = torch.randn(1, 2, 2)

# This one has a shape mismatch on model inputs and outputs:
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 4)
        self.softmax = torch.nn.Softmax()
    def forward(self, x1):
        v4 = x1
        v1 = torch.nn.functional.linear(v4, self.linear.weight, self.linear.bias)
        a1 = torch.tensor([[1, 2], [3, 4]])
        v2 = v1.permute(0, 2, 1)
        v3 = self.softmax(a1)
        return (v1, v2)
# Inputs to the model
x1 = torch.randn(1, 2, 2)

# This one has a shape mismatch on model inputs and outputs:
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 4)
        self.softmax = torch.nn.Softmax()
    def forward(self, x1):
        v4 = x1
        v1 = torch.nn.functional.linear(v4, self.linear.weight, self.linear.bias)
        a1 = torch.tensor([[1, 2], [3, 4]])
        v2 = v1.permute(0, 2, 1)
        v3 = self.softmax(a1)
        return (v1, v3)
# Inputs to the model
x1 = torch.randn(1, 2, 2)

# This one has a shape mismatch on model inputs and outputs:
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 4)
        self.softmax = torch.nn.Softmax()
    def forward(self, x1):
        v4 = x1
        v1 = torch.nn.functional.linear(v4, self.linear.weight, self.linear.bias)
        a1 = torch.tensor([[1, 2], [3, 4]], device='cpu', requires_grad=True)
        v2 = v1.permute(0, 2, 1)
        v3 = self.softmax(a1)
        return (v1, v3)
# Inputs to the model
x1 = torch.randn(1, 2, 2)

# This one has a shape mismatch on model inputs and outputs:
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 4)
        self.softmax = torch.nn.Softmax()
    def forward(self, x1):
        v4 = x1
        v1 = torch.nn.functional.linear(v4, self.linear.weight, self.linear.bias)
        a1 = torch.tensor([[1], [2]], device='cpu', requires_grad=True)
        v2 = v1.permute(0, 2, 1)
        v3 = self.softmax(a1)
        return (v1, v3)
# Inputs to the model
x1 = torch.randn(1, 2, 2)

# This one has a shape mismatch on model inputs and outputs:
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 4)
        self.softmax = torch.nn.Softmax()
    def forward(self, x1):
        v4 = x1
        v1 = torch.nn.functional.linear(v4, self.linear.weight, self.linear.bias)
        a1 = torch.tensor([-1, 3], device='cpu', requires_grad=True)
        v2 = v1.permute(0, 2, 1)
        v3 = self.softmax(a1)
        return (v1, v3)
# Inputs to the model
x1 = torch.randn(1, 2, 2)

# This one has a shape mismatch on model inputs and outputs:
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 4)
        self.softmax = torch.nn.Softmax()
    def forward(self, x1):
        v4 = x1
        v1 = torch.nn.functional.linear(v4, self.linear.weight, self.linear.bias)
        a1 = torch.tensor([[1, 2]], device='cpu', requires_grad=True)
        v2 = v1[:-1, 1:-1, -1].permute(0, 2, 1)
        v3 = self.softmax(a1)
        return (v1, v3)
# Inputs to the model
x1 = torch.randn(1, 4, 2)