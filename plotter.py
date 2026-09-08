import matplotlib.pyplot as plt
import numpy as np

def plot_ilqr(xs, us, dt):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 9))

    M = xs.shape[0]
    N = xs.shape[1] - 1
    t = np.arange(N + 1) * dt

    for i in range(M):
        # color gradient: black (i=0) -> red (i=M-1)
        frac = i / (M - 1) if M > 1 else 1.0
        color = (frac, 0, 0)

        ax1.plot(t, xs[i, :, 0], color=color)
        ax2.plot(t, xs[i, :, 1], color=color)
        ax3.plot(t, us[i, :, 0], color=color)

    ax1.set_title(f'Position over iterations (n={M-1})')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position (rad)')

    ax2.set_title(f'Velocity over iterations (n={M-1})')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (rad/s)')

    ax3.set_title(f'Control over iterations (n={M-1})')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Control (rad/s^2)')

    ax4.axis('off')  # unused subplot

    plt.tight_layout()
    plt.show()