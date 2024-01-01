
c = 32
class Model(torch.nn.Module):
    def __init__(self, dropout_p=0):
        super().__init__()
        self.dropout_p = dropout_p
 
    def forward(self, inputs):
        x, x_mask = inputs[0], inputs[1]
        x = self.drop(x)
        return x
 
# Initializing the model
m = Model()
input_tensor = [[5, 2, 8, 7, 6, 2, 1, 5, 1, 4],
                [2, 9, 1, 1, 1, 1, 9, 4, 8, 9]]
