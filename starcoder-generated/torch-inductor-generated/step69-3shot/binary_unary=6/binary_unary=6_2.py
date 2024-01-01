
class Model(torch.nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.linear = torch.nn.Linear(in_feature, out_feature)
 
    def forward(self, x):
        t1 = self.linear(x)
        t2 = t1 - 2
        t3 = torch.nn.functional.relu(t2)
        return t3

def get_parameters():
    batch_norm_params = torch.nn.BatchNorm2d(3)
    batch_norm_params.weight.data.fill_(0.9)
    batch_norm_params.bias.data.fill_(2)
    return batch_norm_params

# Initializing the model
m = Model(3, 8)

# Use "parameters" instead of "weights".
# Note that the "parameters" field contains
# "batch_norm_params", which is a neural network module.
list(__map__(id, m.parameters(), m.parameters()))

# For example, it may produce the following result:
# [7274211735729, 7274221303264]

