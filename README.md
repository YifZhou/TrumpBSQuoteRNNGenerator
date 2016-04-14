
#The Trump Deep BS Quote RNN Generator

This code implements a simple multi-layer Deep Recurrent Neural Network demonstrating the usage of Tensorflow for training a Trump-like character-level language model based on
my [blog post](http://killianlevacher.github.io/blog/posts/post-2016-03-01/post.html). As a source, the system uses the "wonderful" Trump Tweets collected directly from Mr Trump’s Twitter account
by the [New York Times](http://www.nytimes.com/interactive/2016/01/28/upshot/donald-trump-twitter-insults.html?smid=fb-nytimes&smtyp=cur&_r=1).
Output examples of this model can be also found on this dedicated Trump-like [Twitter-bot](https://twitter.com/TrumpBSQuoteGen).

In a nutshell, this system lets you feed in a set of truly original Trump BS quotes into a neural network, which will produce new unheard of (yet!) Deep Trump BS quotes at your convenience.
If you’re not really in an inspirational mood, the script will let the Trump BS Quote generator produce its own BS quotes all by itself.
If you’re a bit more adventurous however, it also lets you choose a few words to get him started with (eg: ‘Hillary Clinton is a ...’), and finishes the sentence for you in a wonderful Trump-like manner.

The model attempts to learn how to predict the next character in a sequence, eventually learning how to produce the same words and style as the man himself we all love and cherish.

This code is very much inspired by Karpathy's excellent [blog post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) which covers the topic of RNN's in detail (strongly recommended if you're new to that field).
In contrast with Karpathy's code however, which uses pure numpy, the main purpose of this code instead is to get newbies quickly up and running with a basic Deep Learning NLP multi-layer RNN toy starter code,
 using instead the Tensorflow library.

 A detailed description of this code and how to run it can be found on my [blog post](http://killianlevacher.github.io/blog/posts/post-2016-03-01/post.html).

## Usage
**Running the BS Quote Generator using Tensorflow within a Docker Container**
* `docker run --net="host"  -v /Users/Killian1/Development/TrumpBSQuoteRNNGenerator_MasterProject:/src  -it b.gcr.io/tensorflow/tensorflow:0.6.0`




