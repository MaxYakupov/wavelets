import numpy as np
import matplotlib.pyplot as plt

def main():

    plt.figure(1)
    plt.subplot(311)
    plt.title('berlage wavelet f = 10, 20, 40 Hz')
    plt.plot(berlage(fc=10))
    plt.plot(berlage(fc=20))
    plt.plot(berlage(fc=40))

    plt.subplot(312)
    plt.title('puzyrev wavelet f = 10, 20, 40 Hz')
    plt.plot(puzyrev(fc=10))
    plt.plot(puzyrev(fc=20))
    plt.plot(puzyrev(fc=40))

    plt.subplot(313)
    plt.title('rikker wavelet f = 10, 20, 40 Hz')
    plt.plot(ricker(fc=10))
    plt.plot(ricker(fc=20))
    plt.plot(ricker(fc=100))

    plt.tight_layout()
    plt.show()


def berlage(a=1, t= 0.5, n=4, b=5, fc=50, dt=0.001):

    t_arr = np.linspace(start=-t/2+dt, stop=t/2 - dt, num=int(t / dt))
    tn = np.power(t, n)
    exp_bt = np.exp(-b*t_arr)
    cos = np.cos(2*t_arr*fc*np.pi)

    impulse = a*tn*exp_bt*cos

    return impulse


def puzyrev(a=1, len=0.5, shift = 0.25, b = 15, fc=50, dt=0.001):

    t = np.linspace(-shift, len - shift - dt, int(len / dt))
    return a*np.exp(-2*(b*t)**2)*np.cos(2*np.pi*fc*t)


def ricker(fc=10, len=0.5, dt=0.002, peak_loc=0.25):
    """
    from https://github.com/lijunzh/ricker
    ricker creates a shifted causal ricker wavelet (Maxican hat).
    :param fc: center frequency of Ricker wavelet (default 10)
    :param len: float
    :type len: signal length in unit of second (default 0.5 sec)
    :param dt: float
    :type dt: time sampling interval in unit of second (default 0.002 sec)
    :param peak_loc: float
    :type peak_loc: location of wavelet peak in unit of second (default 0.25
    sec)
    :return: shifted Ricker wavelet starting from t=0.
    :rtype: np.ndarray
    Note that the returned signal always starts at t=0. For a different
    starting point, it can be achieved by shifting the time vector instead.
    """
    # Import check
    if fc <= 0:
        raise ValueError("Center frequency (fc) needs to be positive.")

    if len <= 0:
        raise ValueError("Signal length (len) needs to be positive.")

    if dt <= 0:
        raise ValueError("Time interval (dt) needs to be positive.")

    else:
        # Generate time sequence based on sample frequency/period, signal length
        # and peak location
        t = np.linspace(-peak_loc, len - peak_loc - dt, int(len / dt))

        # Shift time to the correct location
        # t_out = t + peak_loc  # time shift Ricker wavelet based on peak_loc

        # Generate Ricker wavelet signal based on reference
        y = (1 - 2 * np.pi ** 2 * fc ** 2 * t ** 2) * np.exp(
            -np.pi ** 2 * fc ** 2 * t ** 2)

    return y
if __name__ == "__main__":
    main()