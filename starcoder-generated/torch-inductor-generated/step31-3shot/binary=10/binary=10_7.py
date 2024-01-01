1
class Model1_0(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 3, bias=False)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 + x2
        return v2

# Model1
class Model1_1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 3, bias=False)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 + x2
        v3 = torch.tanh(v2)
        return v3

# Model1
class Model1_2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 3, bias=False)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 + x2
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m_0 = Model1_0()
m_1 = Model1_1()
m_2 = Model1_2()

# Inputs to the model
x1 = torch.randn(1, 2)
__output__m_0  = m_0(x1, x1)
__output__m_1  = m_1(x1, x1)
__output__m_2  = m_2(x1, x1)

