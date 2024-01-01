
class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, dropout_p=0.0):
        super().__init__()
        self.layers = torch.nn.ModuleList([copy.deepcopy(MyTransformerBlock(input_size,
                                                                hidden_size,
                                                                num_heads,
                                                                dropout_p)) for _ in range(num_layers)])
        
    def forward(self, x1):
        v1 = x1
        for i in range(len(self.layers)):
            v0 = v1
            v1 = self.layers[i](v0)
            v1 = v0 + v1
        return v1
 
# Initializing the model
m = Model(128, 128, 2, 4)

# Inputs to the model
x1 = torch.randn(3, 3, 128)
x2 = torch.randn(3, 3, 128)
x3 = torch.randn(3, 3, 128)
