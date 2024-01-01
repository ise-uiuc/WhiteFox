
class Model(torch.nn.Module):
   def __init__(self):
      super().__init__()

   def forward(self, x1):
       conv2d_1 = torch.nn.Conv2d(3, 16, 7, stride=2, padding=3)
       batch_normalization_1 = torch.nn.BatchNorm2d(16, eps=1e-05)
       relu_1 = torch.nn.ReLU()
       max_pool_2d_1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
       conv2d_2 = torch.nn.Conv2d(16, 32, 5, stride=1, padding=2)
       batch_normalization_2 = torch.nn.BatchNorm2d(32, eps=1e-05)
       relu_2 = torch.nn.ReLU()
       max_pool_2d_2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
       conv2d_3 = torch.nn.Conv2d(32, 16, 3, stride=1, padding=1)
       batch_normalization_3 = torch.nn.BatchNorm2d(16, eps=1e-05)
       relu_3 = torch.nn.ReLU()
       conv2d_4 = torch.nn.Conv2d(16, 8, 3, stride=1, padding=1)
       batch_normalization_4 = torch.nn.BatchNorm2d(8, eps=1e-05)
       relu_4 = torch.nn.ReLU()
       global_max_pooling_2d_1 = torch.nn.AdaptiveMaxPool2d((1, 1))
       reshape_1 = torch.nn.Flatten(start_dim=1, end_dim=-1)
       dropout_1 = torch.nn.Dropout()
       dense_1 = torch.nn.Linear(320, 10)

       v1 = conv2d_1(x1)
       v2 = batch_normalization_1(v1)
       v3 = relu_1(v2)
       v4 = max_pool_2d_1(v3)
       v5 = conv2d_2(v4)
       v6 = batch_normalization_2(v5)
       v7 = relu_2(v6)
       v8 = max_pool_2d_2(v7)
       v9 = conv2d_3(v8)
       v10 = batch_normalization_3(v9)
       v11 = relu_3(v10)
       v12 = conv2d_4(v11)
       v13 = batch_normalization_4(v12)
       v14 = relu_4(v13)
       v15 = global_max_pooling_2d_1(v14)
       v16 = reshape_1(v15)
       v17 = dropout_1(v16)
       v18 = dense_1(v17)

       v19 = torch.softmax(v18, dim=1)

       return v19
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
