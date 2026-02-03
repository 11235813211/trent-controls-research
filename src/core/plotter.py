import matplotlib.pyplot as plt
from typing import List, Tuple
import numpy as np

def plot_outputs(outputs: List[Tuple[float, np.ndarray]], output_names: List[str]):

    n = len(outputs[0][1])
    if n > 12:
        nrows = 4
    if n > 6:
        nrows = 3
    elif n > 3:
        nrows = 2
    else:
        nrows = 1
    
    ncols = int(np.ceil(n / nrows))


    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 5))

    times = np.array([output[0] for output in outputs])
    variables = np.array([output[1] for output in outputs])
    variables = variables.T[0]

    for i, variable in enumerate(variables):
        if nrows > 1:
            axs[i // nrows, i % nrows].plot(times, variable, 'tab:blue')
            axs[i // nrows, i % nrows].set_title(f'{output_names[i]} vs. Time')
        else:
            axs[i // nrows].plot(times, variable, 'tab:blue')
            axs[i // nrows].set_title(f'{output_names[i]} vs. Time')

    plt.show()