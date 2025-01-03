import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CNNLayerNorm(nn.Module):
    def __init__(self,n_feats):
        super(CNNLayerNorm,self).__init__()
        self.layer_norm  = nn.LayerNorm(n_feats)

    def forward(self,x):
        # x (batch, n_freqs, n_times, channels)
        # x (batch, channel, feature, time)
        x = x.transpose(2,3).contiguous() # (batch, channel, time, feature)
        x = self.layer_norm(x)
        x = x.transpose(2,3).contiguous() # (batch, channel, feature, time)
        return x
    
class ResidualBlock(nn.Module):
    """
    Residual using LayerNorm instead of batch norm
    """
    def __init__(self,in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualBlock,self).__init__()
        n_feats=n_feats//2
        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding= kernel//2)
        self.cnn2 = nn.Conv2d(out_channels,out_channels,kernel,stride,padding = kernel//2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self,x):
        residual = x #(batch,channel,feature,time)
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x
    
    
class BidirectionalGRU(nn.Module):
    
    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirectionalGRU,self).__init__()

        self.BiGRU = nn.GRU(
            input_size = rnn_dim, hidden_size = hidden_size,
            num_layers = 1, batch_first = batch_first, bidirectional=True
        )
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x,_ = self.BiGRU(x)
        x = self.dropout(x)
        return x
    
class SleepStateNet(nn.Module):
    
    def __init__(self, n_cnn_layer, n_rnn_layer, rnn_dim, n_class,
                 n_feats, stride=2, dropout=0.1):
        super(SleepStateNet,self).__init__()
        self.cnn = nn.Conv2d(in_channels = 4, out_channels = 128, kernel_size = 3, stride=stride , padding=3//2)

        # n residual cnn layer with filter size of 64
        self.resnet = nn.Sequential(*[ResidualBlock(128, 128, kernel=3, stride=1, dropout=dropout, n_feats=n_feats)
                                     for _ in range(n_cnn_layer)
                                     ])
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,None))
        self.fully_connected = nn.Linear(128,rnn_dim)
        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=rnn_dim if i==0 else rnn_dim*2,hidden_size=rnn_dim,dropout=dropout,batch_first=i==0)
            for i in range(n_rnn_layer)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim*2,rnn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim,n_class)
        )

    def forward(self,x):
        # x (batch, feature, time,channels)
        x = x.permute(0,3,1,2) # x(batch, channels, feature, time)
        x = self.cnn(x)
        x = self.resnet(x)
        x = self.global_avg_pool(x).squeeze(2) # x (batch, channels*feature, time)
        x = x.transpose(1,2) # x (batch,time,channels*feature)
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x

learning_rate = 5e-4
batch_size = 64
epochs = 100
hparams = {
    "n_cnn_layer":5,
    "n_rnn_layer":5,
    "rnn_dim" : 256,
    "n_class" :5,
    "n_feats": 50,
    "stride":2,
    "dropout": 0.3,
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "epochs": epochs
}

def get_model():
    model = SleepStateNet(
        n_cnn_layer = hparams["n_cnn_layer"],
        n_rnn_layer = hparams["n_rnn_layer"],
        rnn_dim = hparams["rnn_dim"],
        n_class = hparams["n_class"],
        n_feats = hparams["n_feats"],
        stride = hparams["stride"],
        dropout = hparams["dropout"]
    )
    return model
