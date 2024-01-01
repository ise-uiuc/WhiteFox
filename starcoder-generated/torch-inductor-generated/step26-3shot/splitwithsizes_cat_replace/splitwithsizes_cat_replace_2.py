
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model_0 = MMDNNModel('mnist_lenet.pb')
        self.model_1 = MMDNNModel('mnist_lenet.pb')
        self.model_2 = MMDNNModel('mnist_lenet.pb')
    def forward(self, v1):
        split_tensors = torch.split(v1, [1], dim=1)
        concatenated_tensor = split_tensors[0]
        return(concatenated_tensor, torch.split(v1, [1], dim=1))
# Input to the model
x = torch.randn(2, 1, 224, 224)
