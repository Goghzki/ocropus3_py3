import torch
from torch import nn
import helpers
from torch.autograd import Variable


def LayoutModel(complexity):
    b = complexity
    r = 3
    model = nn.Sequential(
        nn.Conv2d(1, b, r, padding=r//2),
        nn.BatchNorm2d(b),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(b, b*2, r, padding=r//2),
        nn.BatchNorm2d(b*2),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(b*2, b*4, r, padding=r//2),
        nn.BatchNorm2d(b*4),
        nn.ReLU(),
        LSTM2D(b*4, b*2),
        nn.Conv2d(b*4, b*2, 1),
        nn.BatchNorm2d(b*2),
        nn.ReLU(),
        LSTM2D(b*2, b*2),
        nn.Conv2d(b*4, 1, 1),
        nn.Sigmoid()
    )
    return model


class RowwiseLSTM(nn.Module):
    def __init__(self, ninput=None, noutput=None, ndir=2):
        nn.Module.__init__(self)
        self.ndir = ndir
        self.ninput = ninput
        self.noutput = noutput
        self.bidirectional = True if self.ndir - 1 > 0 else False
        self.lstm = nn.LSTM(ninput, noutput, 1, bidirectional=self.bidirectional)

    def forward(self, img):
        # volatile = not isinstance(img, Variable) or img.volatile
        b, d, h, w = img.size()
        # BDHW -> WHBD -> WB'D
        seq = img.permute(3, 2, 0, 1).contiguous().view(w, h * b, d)
        # WB'D
        h0 = torch.rand(self.ndir, h * b, self.noutput)
        c0 = torch.rand(self.ndir, h * b, self.noutput)
        with torch.no_grad():
            h0 = Variable(h0).cuda()
            c0 = Variable(c0).cuda()
        seqresult, _ = self.lstm(seq, (h0, c0))
        # WB'D' -> BD'HW
        result = seqresult.view(
            w, h, b, self.noutput * self.ndir).permute(2, 3, 1, 0)
        return result


class LSTM2D(nn.Module):
    """A 2D LSTM module."""

    def __init__(self, ninput=None, noutput=None, nhidden=None, ndir=2):
        nn.Module.__init__(self)
        assert ndir in [1, 2]
        nhidden = nhidden or noutput
        self.hlstm = RowwiseLSTM(ninput, nhidden, ndir=ndir)
        self.vlstm = RowwiseLSTM(nhidden * ndir, noutput, ndir=ndir)

    def forward(self, img):
        horiz = self.hlstm(img)
        horizT = horiz.permute(0, 1, 3, 2).contiguous()
        vert = self.vlstm(horizT)
        vertT = vert.permute(0, 1, 3, 2).contiguous()
        return vertT