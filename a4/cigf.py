"""
This is the entry point/ user API for the code.
This code is the implementation of the paper 'From Group to Individual Labels using Deep Features'
which can be found here: http://dkotzias.com/papers/GICF.pdf
(There was a previous version of the paper at a NIPS workshop
http://dkotzias.com/papers/multi-instance%20deep%20learning.pdf)

In order for this to run, data have to be correctly placed and named in a data directory.
For more info on data look at data handler.

For any questions/omissions on this code please contact me. (contact info on the paper)
Thanks for citing :)
                    -Dimitris
"""
from optimize_cost_fn import GICF

if __name__ == '__main__':
    print ('Thanks for reading our paper and using our code.')
    gicf = GICF('amazon')   # 'movies' is the name of one of our datasets
    gicf.set_parameters(batch_size=50, similarity_fn='rbf',lr=0.05,epochs=1,momentum_value=0.8,alpha_balance=0.1,sim_variance=0.7071)    # momentum_value=0.1
    gicf.train()    # this prints results
