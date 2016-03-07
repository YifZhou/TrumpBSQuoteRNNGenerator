
##RUNNING TENSORFLOW WITHIN DOCKER
* Note:  You don't need the docker file it downloads it from the net
* `docker run --net="host"  -v /Users/Killian1/Development/TrumpBSQuoteRNNGenerator_MasterProject:/src  -it b.gcr.io/tensorflow/tensorflow:0.6.0`
 * This downloads a docker image if not already downloaded AND  mounts the local folder to /src AND maps the network ports to local machine
 * __Note:__ I'm purposely putting version 0.6.0 because from 0.7.0 it logs into a jupyter notebook which I don't want (see issues below)

## RUNNING TENSORBOARD
1. Make sure you're not on the proxy otherwise you won't be able to access the tensorboard URL
2. Within the docker environment, run `tensorboard --logdir=/path/to/log-directory`
 * Path to dir points to the dir where the tensorboard files collected during the last run of the program are located
 * That path needs to be from the context of the docker environment of course
3. Then go to the URL mentioned and replace localhost with the docker-machine IP address

* Why you can't see the values of each Tensor UNLESS you do a session.eval
 * https://stackoverflow.com/questions/33633370/how-to-print-the-value-of-a-tensor-object-in-tensorflow


## CURRENT ISSUES TO SOLVE

#### ISSUE TITLE TEMPLATE
- __Problem:__ blablabla
- __Solution:__ blablabla
- __Notes:__
*  blablabla


#### TENSORFLOW DOCKER 0.7.0 GOES STRAIGHT TO THE JUPYTER NOTEBOOK
- __Problem:__ for docker 0.6.0 you would go in the terminal without automatically getting into jupyter which I don't want.
Need to find a way to stop the notebook without going out of environment or else maybe I can't because the entire docker is a jupyter notebook? In which case I would have to create my own docker file perhaps.
- __Solution:__ blablabla
- __Notes:__
*  blablabla


## SOLVED ISSUES

#### SSL CERTIFICATE ISSUE WHEN INSTALLING TENSORFLOW
- __Problem:__  it was complaining about a certificate when trying to install tensorflow
sudo  pip install --upgrade https://storage.googleapis.com/tensorflow/mac/tensorflow-0.7.1-cp27-none-any.whl
SSLError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed (_ssl.c:590)
- __Solution:__ I got it resolved by downloading the .whl file locally and then installing tensor flow from my machine
- __Notes:__


#### ADDING TENSORBOARD SUMMARY CAUSES ERROR WHEN FEEDING IN DATA TO VALIDATION MODEL
- __Problem:__ Was getting exception of type InvalidArgurment Exception
- __Solution:__ The reason is because the validation tensorboard summary variables I had defined where defined on the training model graph, hence it was the training cost variable was pointing to the training graph for which I was not feeding in inputs since I was feeing inputs to the validation model instead
- __Notes:__
 * Describes the issue with the problem and solution: http://stackoverflow.com/questions/35114376/error-when-computing-summaries-in-tensorflow
 * Very similar to this issue if not identical: https://groups.google.com/a/tensorflow.org/forum/#!topic/discuss/oFwmT_d42oU
 * When you remove the tensorboard variables in the session.run it all works fine
 * http://stackoverflow.com/questions/35114376/error-when-computing-summaries-in-tensorflow