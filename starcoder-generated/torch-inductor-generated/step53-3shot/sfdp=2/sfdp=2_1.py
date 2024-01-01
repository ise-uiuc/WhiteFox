
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_layer = MultiHeadAttentionLayer(10, 21)
 
    def forward(self, x):
        multi_head_output = self.attention_layer(x)
        return multi_head_output

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 2, 10, 21)
