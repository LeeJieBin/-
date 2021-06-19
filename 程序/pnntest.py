import sys
sys.path.append('D:\\123\\draw\\PlotNeuralNet-master')
import pycore.tikzeng as tik
from pycore.tikzeng import *

# defined your arch
arch = [
  to_head( '..' ),
  to_cor(),
  to_begin(),
  to_input('D:/BaiduNetdiskDownload/rock_all_data/Data_all/55_0.jpg',width=8, height=8),
  to_Conv("conv1", 112, 32, offset="(0,0,0)", to="(0,0,0)", height=64, depth=64, width=2 ),
  to_Pool("pool1", offset="(1,0,0)", to="(conv1-east)", height=32, depth=32, width=1),
  to_connection( "conv1", "pool1"),
  to_Conv("conv2", 56, 32, offset="(2,0,0)", to="(pool1-east)", height=32, depth=32, width=2 ),
  to_connection( "pool1", "conv2"),
  to_Pool("pool2", offset="(1,0,0)", to="(conv2-east)", height=28, depth=28, width=1),
  to_connection( "conv2", "pool2"),
  to_Conv("conv3", 54, 64, offset="(5,0,0)", to="(pool1-east)", height=28, depth=28, width=4 ),
  to_connection( "pool2", "conv3"),
  to_Conv("conv4", 54, 64, offset="(7,0,0)", to="(pool1-east)", height=28, depth=28, width=4 ),
  to_connection( "conv3", "conv4"),
  to_Pool("pool3", offset="(6,0,0)", to="(conv2-east)", height=24, depth=24, width=1),
  to_connection( "conv4", "pool3"),
  to_SoftMax("soft1", 7 ,"(10,0,0)", "(pool1-east)", caption="SOFT"  ),
  to_connection("pool3", "soft1"),
  to_end()
  ]

def main():
  namefile = str(sys.argv[0]).split('.')[0]
  to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
  main()
  
#D:/123/draw/PlotNeuralNet-master/layers/