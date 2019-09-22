# Caffe_Classification
How to use Caffe model to classify images

## Setup
import numpy and matplotlib

## load caffe
sys.path.insert(0, caffe_root + 'python')

## load model
caffe.set_mode_cpu()

#### define structure and extract weights
model_def = caffe_root + '[deploy.prototxt]'
model_weights = caffe_root + '[caffemodel]'

net = caffe.Net(model_def, model_weights, caffe.TEST) # use test mode

## image preprocessing
  matplotlib will load images with values in the range [0,1] in RGB format, innermost
  CaffeNet takes images in BGR format in the range of [0,255] and has the mean imageNet pixel value subtracted from them, outermost
  
#### load the mean ImageNet image
mu = np.load(caffe_root + '[mean.npy]'
#### average over pixels to obtain the mean pixel values
mu = mu.mean(1).mean(1)
#### create transformer
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
#### set transformer

## Perform classification
#### set blobs (input_size)
#### load an image and transform it
#### copy the image data into the memory allocated for the net
#### perform classification
net.forward()
#### load ImageNet labels
print ('output label:', labels[output_prob.argmax()])

## Sort top five predictions
output_prob.argsort()[::-1][:5] # default ascending
zip(output_prob[top_inds], labels[top_inds])

## Switch to GPU
%timeit net.forward() # cpu time
caffe.net_device(0)
caffe.set_mode_gpu()
net.forward()
%timeit net.forwad() # gpu time
