
class __ModelName__(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.__linear_1 = torch.nn.Linear(3, 9, bias=True)
        self.__linear_2 = torch.nn.Linear(3, 32, bias=True)
        self.__linear_3 = torch.nn.Linear(3, 2, bias=True)
 
    def forward(self, x5):
        v1 = torch.nn.functional.silu(self.__linear_1(x5))
        v2 = v1 * 0.5
        v3 = (v1 * v1) * v1
        v3 = v3 * 0.044714998277471
        v4 = v3 * 0.7978845608028654
        v5 = torch.tanh(v4)
        v6 = v5 + 1
        v7 = v2 * v6
        v8 = torch.nn.functional.silu(self.__linear_2(x5))
        v9 = v8 * 0.5
        v10 = (v8 * v8) * v8
        v10 = v10 * 0.044714998277471
        v11 = v10 * 0.7978845608028654
        v12 = torch.tanh(v11)
        v13 = v12 + 1
        v14 = v9 * v13
        v15 = torch.softmax(self.__linear_3(x5), dim=1)
        v16 = v7 + v14
        v17 = (v16 * v15)
        return v17

# Initializing the model
m = __ModelName__()

# Inputs to the model
x5 = torch.randn(1, 3)
