from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange
# we should download the 'past' module by 'pip install future' ????!!!!

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]

        # print("X[i].shape = ", X[i].shape)
        # print(f'scores = {scores}, correct_class_score = {correct_class_score}')

        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin

                dW[:, j] += X[i].T # W's j-th column += X[i]^T
                dW[:, y[i]] -= X[i].T # W's y_i-th column -= X[i]^T

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W) # W * W : element-wise multiplication
    
    dW /= num_train
    dW += 2 * reg * W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather than first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # pass
    
    # W * W : element-wise multiplication
    # W : D * C, X[i] : D * 1, score[i] = W^T * X[i] : C * 1

    # loss = 1/N * sum_{i}Li + reg * sum(W * W)
    # Li = \sum_{j=1,j!=yi}^C max(0, score[j] - score[yi] + delta)
    # Li = \sum_{j=1,j!=yi}^C max(0, s[i,j] - S[i,yi] + delta)

    # S: score N * C, S = X^T * W
    # S[i,j] : wj.dot(xi)
    # wj: W's j-th column , xi: X's i-th row

    # dloss/dW_{k,m} = 1/N * sum_{i=1}^N dLi/dW_{k,m} + reg * 2 * W_{k,m}
    # score[j] - score[y[i]] + delta > 0:
    # W's j-th column += X[i]^T
    # W's y_i-th column -= X[i]^T

    # dW = 1 / n * (sum d margin/dW + reg * 2 * W) 

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW

def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # pass

    N = X.shape[0]

    S = X.dot(W) # N * C
    # S[i,j] : xi.dot(wj) X's i-th row, W's j-th column

    delta = 1
    # print("np.arrange(N) = ", np.arange(N))
    S_correct = S[np.arange(N), y].reshape(-1, 1) # N * 1
    margin = np.maximum(0, S - S_correct + delta) # N * C

    N = X.shape[0]
    loss = np.sum(margin) / N - delta
    loss += reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # pass

    margin[margin > 0] = 1
    mask = np.array(margin)
    mask[np.arange(N), y] -= np.sum(mask, axis=1)
    dW = X.T.dot(mask) / N
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
