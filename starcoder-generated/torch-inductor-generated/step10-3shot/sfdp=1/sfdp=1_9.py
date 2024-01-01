
class Model(torch.nn.Module):
    def __init__(self, hidden_size, intermediate_size, activation):
        super().__init__()
        self.dropout = torch.nn.Dropout(self.__constants__.get("dropout_p", 0.0))
        self.activation = self._wrap_activation(activation)
        self.intermediate_dense = torch.nn.Linear(self.__constants__.get("hidden_size", hidden_size), self.__constants__.get("intermediate_size", intermediate_size))
 
    def forward(self, x3):
        v11 = torch.matmul(x3, x3.transpose(-2, -1))
        v21 = v11.div(0.0702)
        v31 = v21.softmax(dim=-1)
        v41 = self.dropout(v31)
        v51 = torch.matmul(v41, x3)
        v61 = self.intermediate_dense(v51)
        v71 = self.activation(v61)
        return v71

# Initializing the model
m = Model(hidden_size=64, intermediate_size=32, activation=torch.nn.ReLU)

# Inputs to the model
x3 = torch.randn(2, 5, 64)
