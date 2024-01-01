
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout1 = torch.nn.Dropout(p=0.2)
        self.dropout2 = torch.nn.Dropout(p=0.2)
    def forward(self, x1):
        x2 = self.dropout1(x1)
        x3 = self.dropout2(x1)
        return (x2, x3)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
# Model end

# Model begins
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout1 = F.dropout
        self.dropout2 = F.dropout
    def forward(self, x1):
        x2 = self.dropout1(x1)
        x3 = self.dropout2(x1)
        return (x2, x3)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
# Model end

# Model begins
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout1 = torch.nn.functional.dropout
        self.dropout2 = torch.nn.functional.dropout
    def forward(self, x1):
        x2 = self.dropout1(x1)
        x3 = self.dropout2(x1)
        return (x2, x3)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
# Model end

# Model begins
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout1 = F.drop
        self.dropout2 = F.drop
    def forward(self, x1):
        x2 = self.dropout1(x1)
        x3 = self.dropout2(x1)
        return (x2, x3)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
# Model end

# Model begins
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout1 = torch.nn.drop
        self.dropout2 = torch.nn.drop
    def forward(self, x1):
        x2 = self.dropout1(x1)
        x3 = self.dropout2(x1)
        return (x2, x3)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
# Model end

# Model begins
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout1 = torch.nn.functional.drop
        self.dropout2 = torch.nn.functional.drop
    def forward(self, x1, x2):
        x3 = self.dropout1(x1, x2)
        x4 = self.dropout2(x1, x2)
        return (x3, x4)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
