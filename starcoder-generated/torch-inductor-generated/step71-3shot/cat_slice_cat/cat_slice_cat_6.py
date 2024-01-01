
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
  
    def forward(self, x1):
        list1 = [x1[:, 0, 0], x1[:, 1, 1], x1[:, 2, 2]]
        t1 = torch.cat(list1, 1)
        t2 = t1[:, 0:9223372036854775807]
        t3 = t2[:, 0:size]
        list2 = [t1, t3]
        t4 = torch.cat(list2, 1)
        return t4

# Initializing the model
m = Model()

# Input to the model
torch.manual_seed(1337)
x1 = torch.randn(100, 3, 64, 128)
