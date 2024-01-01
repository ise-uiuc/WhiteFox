
class _Input(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = None
        self.bn = None
        self.act = None
 
    def forward(self, input_1):
        v1 = self.conv(input_1)
        if self.bn is not None:
            v1 = self.bn(v1)
        if self.act is not None:
            v1 = self.act(v1)
        return v1

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_features=299, out_features=17)
        self.linear2 = torch.nn.Linear(in_features=17, out_features=17)
 
    def forward(self, input_1):
        v1 = _Input()
        v2 = input_1
        v2.__setattr__("conv", lambda x: self.linear1)
        v2.__setattr__("out_channels", 17)
        v3 = _Input()
        v3.__setattr__("conv", lambda x: self.linear2)
        v3.__setattr__("linear2", self.linear2)
        v3.__setattr__("other", self.linear1.weight.data)
        x1 = v1(v2)
        x2 = v3(x1)
        return x2

# Initializing the model

class _Input(torch.nn.Module):
     def __init__(self):
        super().__init__()
        self.conv = None
        self.bn = None
        self.act = None
 
     def forward(self, input_1):
         v1 = self.conv(input_1)
         if self.bn is not None:
             v1 = self.bn(v1)
         if self.act is not None:
             v1 = self.act(v1)
         return v1
 
m = Model()
