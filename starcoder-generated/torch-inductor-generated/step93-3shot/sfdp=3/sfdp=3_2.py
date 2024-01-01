
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, q, k, v, input_mask=None):
        output = q @ k.transpose(-2, -1)
        output = output * scale_factor
        output = torch.nn.functional.softmax(output, dim=-1)
        if input_mask!= None:
            output.masked_fill_(input_mask, 0)
        output = self.dropout(output)
        output = output @ v

# Initializing the model
m = Model()

# Inputs to the model
output = m(x, y, z)

# Weights to be fine tuned
__weights_dict__ = dict()
__weights_dict__["x"] = torch.tensor()
__weights_dict__["y"] = torch.tensor()
__weights_dict__["z"] = torch.tensor()
__weights_dict__["scale_factor"] = torch.tensor()
__weights_dict__["dropout_p"] = torch.tensor()

torch.save(__weights_dict__, "model.pth")

