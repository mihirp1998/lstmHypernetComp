import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from modules import ConvLSTMCell,ConvLSTMCellTemp, Sign
import numpy as np
from  utils import batchConv2d,BatchConv2dContext

class EncoderCell(nn.Module):
    def __init__(self):
        super(EncoderCell, self).__init__()

        # self.conv = nn.Conv2d(
        #     3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv = BatchConv2dContext(3,64,kernel_size=3,stride=2, padding=1, bias=False)

        #self.hyper1 = HyperConvLSTMCell(64,256,256,128,stride=2)
        self.rnn1 = ConvLSTMCellTemp(
            64,
            256,
            kernel_size=3,
            stride=2,
            padding=1,
            hidden_kernel_size=1,
            bias=False)
        self.rnn2 = ConvLSTMCellTemp(
            256,
            512,
            kernel_size=3,
            stride=2,
            padding=1,
            hidden_kernel_size=1,
            bias=False)
        self.rnn3 = ConvLSTMCellTemp(
            512,
            512,
            kernel_size=3,
            stride=2,
            padding=1,
            hidden_kernel_size=1,
            bias=False)

    def forward(self, input,context,encoder_h,enc_hyper_h,batchsize):
        self.batchsize=batchsize

        hidden1,hidden2,hidden3 =  encoder_h
        hyperhidden1,hyperhidden3,hyperhidden3 = enc_hyper_h 

        x = self.conv(input,context,batchsize)

        hidden1,hyperhidden1 = self.rnn1(x,context,hidden1,hyperhidden1,self.batchsize)
        x = hidden1[0]
        context_x = hyperhidden1[0]

        hidden2,,hyperhidden2 = self.rnn2(x,context_x,hidden2,hyperhidden2,self.batchsize)
        x = hidden2[0]
        context_x = hyperhidden2[0]

        hidden3,,hyperhidden3 = self.rnn3(x,context_x,hidden3,hyperhidden3,self.batchsize)
        x = hidden3[0]
        context_x = hyperhidden3[0]

        return x,context_x,[hidden1, hidden2, hidden3],[hyperhidden1,hyperhidden2,hyperhidden3]


class Binarizer(nn.Module):
    def __init__(self):
        super(Binarizer, self).__init__()
        #self.conv = nn.Conv2d(512, 32, kernel_size=1, bias=False)
        self.conv = BatchConv2dContext(512,32,kernel_size=1,stride=1, padding=0, bias=False)
        self.sign = Sign()

    def forward(self, input,context,batchsize):
        feat = self.conv(input,context,batchsize)
        # feat = batchConv2d(input,context,batchsize,stride=1, padding=0, bias=False)
        x = F.tanh(feat)
        return self.sign(x)


class HyperNetwork(nn.Module):

    def __init__(self,num_vids):
        super(HyperNetwork, self).__init__()
        emb_size = num_vids
        emb_dimension= 16

        self.context_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=False)
        initrange = 0.5 / emb_dimension
        self.context_embeddings.weight.data.uniform_(-initrange, initrange)
        # self.z_dim = z_dim
        # enclayer =   [64*3*3*3]+[1024*64*3*3]+[1024*256*1*1]+[2048*256*3*3]+[2048*512*1*1]+[2048*512*3*3]+[2048*512*1*1]
        # declayer = [512*32*1*1]+[2048*512*3*3] + [2048*512*1*1] +[2048*128*3*3] + [2048*512*1*1] + [1024*128*3*3] + [1024*256*3*3] + [512*64*3*3] + [512*128*3*3] + [32*3*1*1]
        # binlayer=[512*32*1*1]
        # layer = np.array(declayer+enclayer+binlayer)
        # #total is 18333792
        # self.layer_cum= np.cumsum(layer) 
        # #self.layer_cum= np.cumsum(declayer) 

        # f = self.layer_cum[-1]
        # #f=2359296
        # # out = self.out_size*self.f_size*self.f_size*self.in_size
        # self.w1 = Parameter(torch.fmod(torch.randn((emb_dimension, f)),2))
        # self.b1 = Parameter(torch.fmod(torch.zeros((f)),2))

        # self.w2 = Parameter(torch.fmod(torch.randn((h,f)),2))
        # self.b2 = Parameter(torch.fmod(torch.randn((f)),2))

    def forward(self,id_num,batchsize):
        self.batchsize= batchsize
        contextEmbed = self.context_embeddings(id_num)
        return contextEmbed
        # h_final = torch.matmul(contextEmbed, self.w1) + self.b1

        # dec_init_conv = h_final[:,:self.layer_cum[0]]
        # dec_init_conv = dec_init_conv.view(self.batchsize,512,32,1,1)
        # #print("datatype",init_conv.dtype)
        # dec_rnn1_i = h_final[:,self.layer_cum[0]:self.layer_cum[1]]
        # #print(rnn1_i.shape)
        # dec_rnn1_i = dec_rnn1_i.view(self.batchsize,2048,512,3,3)
        
        # dec_rnn1_h = h_final[:,self.layer_cum[1]:self.layer_cum[2]]
        # dec_rnn1_h = dec_rnn1_h.view(self.batchsize,2048,512,1,1)

        # dec_rnn2_i = h_final[:,self.layer_cum[2]:self.layer_cum[3]]
        # dec_rnn2_i = dec_rnn2_i.view(self.batchsize,2048,128,3,3)

        # dec_rnn2_h = h_final[:,self.layer_cum[3]:self.layer_cum[4]]
        # dec_rnn2_h = dec_rnn2_h.view(self.batchsize,2048,512,1,1)

        # dec_rnn3_i = h_final[:,self.layer_cum[4]:self.layer_cum[5]]
        # dec_rnn3_i = dec_rnn3_i.view(self.batchsize,1024,128,3,3)

        # dec_rnn3_h = h_final[:,self.layer_cum[5]:self.layer_cum[6]]
        # dec_rnn3_h = dec_rnn3_h.view(self.batchsize,1024,256,3,3)

        # dec_rnn4_i = h_final[:,self.layer_cum[6]:self.layer_cum[7]]
        # dec_rnn4_i = dec_rnn4_i.view(self.batchsize,512,64,3,3)

        # dec_rnn4_h = h_final[:,self.layer_cum[7]:self.layer_cum[8]]
        # dec_rnn4_h = dec_rnn4_h.view(self.batchsize,512,128,3,3)

        # dec_final_conv = h_final[:,self.layer_cum[8]:self.layer_cum[9]]
        # dec_final_conv = dec_final_conv.view(self.batchsize,3,32,1,1)



        # enc_init_conv = h_final[:,self.layer_cum[9]:self.layer_cum[10]]
        # enc_init_conv = enc_init_conv.view(self.batchsize,64,3,3,3)

        # enc_rnn1_i = h_final[:,self.layer_cum[10]:self.layer_cum[11]]
        # enc_rnn1_i = enc_rnn1_i.view(self.batchsize,1024,64,3,3)

        # enc_rnn1_h = h_final[:,self.layer_cum[11]:self.layer_cum[12]]
        # enc_rnn1_h = enc_rnn1_h.view(self.batchsize,1024,256,1,1)

        # enc_rnn2_i = h_final[:,self.layer_cum[12]:self.layer_cum[13]]
        # enc_rnn2_i = enc_rnn2_i.view(self.batchsize,2048,256,3,3)

        # enc_rnn2_h = h_final[:,self.layer_cum[13]:self.layer_cum[14]]
        # enc_rnn2_h = enc_rnn2_h.view(self.batchsize,2048,512,1,1)

        # enc_rnn3_i = h_final[:,self.layer_cum[14]:self.layer_cum[15]]
        # enc_rnn3_i = enc_rnn3_i.view(self.batchsize,2048,512,3,3)

        # enc_rnn3_h = h_final[:,self.layer_cum[15]:self.layer_cum[16]]
        # enc_rnn3_h = enc_rnn3_h.view(self.batchsize,2048,512,1,1)
        

        # bin_init_conv = h_final[:,self.layer_cum[16]:self.layer_cum[17]]
        # bin_init_conv = bin_init_conv.view(self.batchsize,32,512,1,1)


        # return [[enc_init_conv,enc_rnn1_i,enc_rnn1_h,enc_rnn2_i,enc_rnn2_h,enc_rnn3_i,enc_rnn3_h],[dec_init_conv,dec_rnn1_i,dec_rnn1_h,dec_rnn2_i,dec_rnn2_h,dec_rnn3_i,dec_rnn3_h,dec_rnn4_i,dec_rnn4_h,dec_final_conv],bin_init_conv]



class DecoderCell(nn.Module):
    def __init__(self):
        super(DecoderCell, self).__init__()
        
        ''' self.conv1 = nn.Conv2d(
            32, 512, kernel_size=1, stride=1, padding=0, bias=False)'''
        self.conv1 = BatchConv2dContext(32, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.rnn1 = ConvLSTMCellTemp(
            512,
            512,
            kernel_size=3,
            stride=1,
            padding=1,
            hidden_kernel_size=1,
            bias=False)
        self.rnn2 = ConvLSTMCellTemp(
            128,
            512,
            kernel_size=3,
            stride=1,
            padding=1,
            hidden_kernel_size=1,
            bias=False)
        self.rnn3 = ConvLSTMCellTemp(
            128,
            256,
            kernel_size=3,
            stride=1,
            padding=1,
            hidden_kernel_size=3,
            bias=False)
        self.rnn4 = ConvLSTMCellTemp(
            64,
            128,
            kernel_size=3,
            stride=1,
            padding=1,
            hidden_kernel_size=3,
            bias=False)
        self.conv2 = BatchConv2dContext(32, 3, kernel_size=1, stride=1, padding=0, bias=False)

        '''
        self.conv2 = nn.Conv2d(
            32, 3, kernel_size=1, stride=1, padding=0, bias=False)
        '''


    def forward(self, input,context, decoder_h,dec_hyper_h,batchsize):
        self.batchsize=batchsize

        hidden1,hidden2,hidden3,hidden4 =  decoder_h
        hyperhidden1,hyperhidden3,hyperhidden3,hyperhidden4 = dec_hyper_h

        x = self.conv(input,context,batchsize)

        hidden1,hyperhidden1 = self.rnn1(x,context,hidden1,hyperhidden1,self.batchsize)
        x = hidden1[0]
        context_x = hyperhidden1[0]
        x = F.pixel_shuffle(x, 2)

        hidden2,hyperhidden2 = self.rnn2(x,context_x,hidden2,hyperhidden2,self.batchsize)
        x = hidden2[0]
        context_x = hyperhidden2[0]
        x = F.pixel_shuffle(x, 2)

        hidden3,hyperhidden3 = self.rnn3(x,context_x,hidden3,hyperhidden3,self.batchsize)
        x = hidden3[0]
        context_x = hyperhidden3[0]
        x = F.pixel_shuffle(x, 2)

        hidden4,hyperhidden4 = self.rnn4(x,context_x,hidden4,hyperhidden4,self.batchsize)
        x = hidden4[0]
        context_x = hyperhidden4[0]
        x = F.pixel_shuffle(x, 2)

        x = self.conv2(x,context_x,batchsize)
        x = F.tanh(x) / 2

        return x, context_x,[hidden1, hidden2, hidden3, hidden4],[hyperhidden1,hyperhidden2,hyperhidden3,hyperhidden4]