import time
import random as pyrandom
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

def relu(x):
    return jnp.maximum(0, x)

def predict(params, input):
    # per-example predictions
    activations = input
    (w1, b1), (w2, b2) = params
    o1 = relu(jnp.dot(w1, activations) + b1)
    o2 = jnp.dot(w2, jnp.concatenate((o1, input))) + b2
    return o2
batched_predict = vmap(predict, in_axes=(None, 0))

def init_handpicked():
    w1 = jnp.array([[1.0, -1], [-1.0, 1]])
    b1 = jnp.array([0.0,0])
    w2 = jnp.array([[.5, .5, .5, .5]])
    b2 = jnp.array([0.0])
    return ((w1, b1), (w2, b2))

def init_random(key):
    w1 = random.normal(key, (2,2), )
    b1 = random.normal(key, (2,))
    w2 = random.uniform(key, (1,4))
    b2 = random.uniform(key, (1,))
    return ((w1, b1), (w2, b2))

def loss(params, inputs, targets):
    preds = jnp.concatenate(batched_predict(params, inputs))
    return jnp.linalg.norm(preds - targets)

def strparams(params):
    (w1, b1), (w2, b2) = params
    return "\n".join([str(p) for p in (w1,b1,w2,b2)])


def update(params, inputs, targets, step_size):
    preds = jnp.concatenate(batched_predict(params, inputs))
    grads = grad(loss)(params, inputs, targets)
    new_params = [(w - step_size * dw, b - step_size * db) for (w, b), (dw, db) in zip(params, grads)]

    return new_params


def main():
    seed = pyrandom.randint(0, 1000)
    print("Seed {}".format(seed))
    key = random.PRNGKey(seed)
    # params = init_handpicked()
    params = init_random(key)
    print("Initial params")
    print(strparams(params))
    step_size = .00001
    for epoch in range(1000):
        # start_time = time.time()
        key, subkey = random.split(key)
        train_inputs = random.uniform(subkey, (90,2), minval=-100, maxval=100)
        train_targets = jnp.max(train_inputs, axis=1)
        # epoch_time = time.time() - start_time
        # print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
        key, subkey = random.split(subkey)
        test_inputs = random.uniform(subkey, (10,2),  minval=-100, maxval=100)
        test_targets = jnp.max(test_inputs, axis=1)
        # train_loss = loss(params, train_inputs, train_targets)
        # print("Training set loss {}".format(train_loss))
        test_loss = loss(params, test_inputs, test_targets)
        print("Test set loss {}".format(test_loss))
        params = update(params, train_inputs, train_targets, step_size)
    print("Final params")
    print(strparams(params))
    key, subkey = random.split(subkey)
    test_inputs = random.uniform(subkey, (10,2), minval=-100, maxval=100)
    test_targets = jnp.max(test_inputs, axis=1)
    test_loss = loss(params, test_inputs, test_targets)
    print("Final test set predictions")
    print(test_inputs)
    print(jnp.concatenate(batched_predict(params, test_inputs)))
    print("Final test set loss {}".format(test_loss))




main()
