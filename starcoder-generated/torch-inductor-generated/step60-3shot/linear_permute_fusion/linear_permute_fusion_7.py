
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.lstmcell = torch.nn.LSTMCell(2, 2)
    def forward(self, x3):
        v3 = torch.nn.functional.linear(x3, self.linear.weight, self.linear.bias)
        v4 = v3.permute(0, 2, 1)
        v5 = self.lstmcell(v4)
        v6 = torch.nn.functional.linear(v4, self.lstmcell.weight_ih, self.lstmcell.bias_ih) # linear1 with ih weights and biases
        v7 = v6.permute(0, 2, 1)
        v8 = torch.nn.functional.linear(v4, self.lstmcell.weight_hh, self.lstmcell.bias_hh) # linear2 with hh weights and biases
        v9 = v8.permute(0, 2, 1)
        v10 = v7 + v9
        v11 = v7.clone()
        v12 = v11 + v10
        v13 = torch.nn.functional.linear(v12, self.lstmcell.weight_ih, self.lstmcell.bias_ih) # linear3 with ih weights and biases
        v14 = v13.permute(0, 2, 1)
        v15 = torch.nn.functional.linear(v12, self.lstmcell.weight_hh, self.lstmcell.bias_hh) # linear4 with hh weights and biases
        v16 = v15.permute(0, 2, 1)
        v17 = v14 + v16
        v18 = v14.clone()
        v19 = v18 + v17
        v20 = v18.flip(1)
        return v20, v0
# Inputs to the model
x3 = torch.randn(1, 2, 2)
