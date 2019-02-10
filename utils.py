import torch.nn.functional as F

def batchConv2d(layer,kernel,batchsize,stride,padding,bias,dilation=1):
    out_dim = kernel.shape[1]
    kernel = kernel.contiguous().view([-1]+list(kernel.shape[2:]))
     # mixing batch size and output dim
    layer =layer.view([1,-1] + list(layer.shape[2:]))
     # mixing batch size and input dim
    #print(layer.shape,kernel.shape,"check") 
    layer = F.conv2d(layer, kernel,groups=batchsize,stride=stride,padding=padding,dilation=dilation)
    layer= layer.view(batchsize,out_dim,layer.shape[2],layer.shape[3])
    #unsqueezing the layer
    return layer

class BatchConv2dContext:
    def __init__(self,in_channel,out_channel,kernel_size,stride,padding,bias):
        super(BatchConv2dContext, self).__init__()
        ksize = in_channel * out_channel * kernel_size * kernel_size 
        self.hyper2KernelW_i = Parameter(torch.fmod(torch.randn((16, ksize)),2))
        self.hyper2KernelB_i = Parameter(torch.fmod(torch.randn(ksize),2))
        self.stride= stride
        self.padding= padding 
        self.bias = bias
        self.dilation=1

    def Conv2dContext(layer,context,batchsize):
        kernel = torch.matmul(context, self.hyper2KernelW_i) + self.hyper2KernelB_i
        layer = batchConv2d(layer,kernel,batchsize,self.stride,self.padding,self.bias)
        return layer    