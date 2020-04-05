# girl

Ha! I am implmenting my own girl, which is a collections of RL algorithms starting from bandit algorithm.

## Advantages

- This code base is aimed at exhibiting the algorithm in RL field. Instead of those computation efficiency tricks.

- This will make it easier to read and learn for almost any level of users. (But I am not going to write a document to teach you because I plan to make you learn everything directly from code)

- Different from [rlpyt](https://github.com/asstooke/rlpyt) which is a fantastic implementation, you can see the interfaces in all `base` file and implement as you need.

- Also, the repo will use [pytorch](https://pytorch.org/) which I think is a lot more readable than tensorflow.

## Code Conventions and Orgnizations

- All computations are mostly used in numpy ndarrays. Only when data goes into agent or algorithm, it will be Pytorch tensors by default.

    - Something in the middle of the program (e.g. replay buffer, or agent-env interacting) will be added with `_pyt` postfix for torch.Tensor.

    - All array given to agent will be in batch-wise, for the convention of pytorch.

- Each `Base` class defines the necessary interface of implementing the components, with `NotImplementedError` means you have to implement this method to make the class work.

    All interface described in `Base` classes should be meet. Incase some implememntation need more arguments, which has to be keyword arguments adding at the end.

- `concepts` is a directory specifying some of the necessary RL concepts, usually defines numerically, but does not specify the type of array.

- `experiments`, `sampler` and `runner` are aimed to make the code modular, which does not involve RL concepts

- `utils` is a directory specifying some tools of running the code, they are not directly linked with RL concepts.

## Installation

1. This package inherit lots of code from [rlpyt](https://github.com/ZiwenZhuang/rlpyt). So, if you need checking dependencies, please go and have a look. You can also take the `environment.yml` if you are lazy, but it is not garanteed to work.

2. Install [exptools](https://github.com/ZiwenZhuang/exptools) to run experiment and visualize curves.

3. Install this repo using `pip install -e .` And do notice install in the exact conda environment you are using.
