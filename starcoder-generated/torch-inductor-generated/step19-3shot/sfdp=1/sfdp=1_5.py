
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor, scaled_factor, dropout_p):
        intermediate = input_tensor.matmul(scaled_factor.transpose(-2, -1))
        intermediate = intermediate.div(inv_scale_factor)
        softMaxIntermediate = torch.nn.functional.softmax(intermediate, dim=-1)
        dropoutIntermediate = torch.nn.functional.dropout(softMaxIntermediate, p=dropout_p)
        return dropoutIntermediate.matmul(input_tensor)
# Initializing the model
m = Model()
input_tensor = torch.randn(3, 5, 6)
scaled_factor = torch.randn(7, 5, 3)
dropout_p = 0.1

