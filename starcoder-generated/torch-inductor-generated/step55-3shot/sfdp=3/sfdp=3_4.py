
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.attention_dropout = 0.0
 
    def forward(self, __input__):
        # __input__ shape: [batch_size, nb_heads, sequence_length, nb_head_features]
        # output shape: [batch_size, nb_heads, sequence_length, nb_head_features]
        return output

# Initializing the model
m = Model()

# Inputs to the model
