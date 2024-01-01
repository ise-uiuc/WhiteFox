
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
    def forward(self, x1):
        l1 = torch.matmul(x1, self.w)
        l2 = l1 + self.b
        l3 = torch.clamp_min(l2, 0.)
        l4 = torch.clamp_max(l3, 6.)
        l5 = l4 / 6.
        return l5


# Initializing the model
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.w = torch.randn(20, 30)
        self.b = torch.randn(30)
 
    def forward(self, inputTensor):
        l1 = torch.matmul(inputTensor, self.w)
        l2 = l1 + self.b
        l3 = torch.clamp_min(l2, 0.)
        l4 = torch.clamp_max(l3, 6.)
        l5 = l4 / 6.
        return l5


# Inputs to the model
x1 = torch.randn(32, 30)

