import matplotlib.pyplot as plt
import numpy as np


# Let delta_{t,i}=nb share at time t for t = 0,..,T, and each trajectory i = 1,...,N.
# return = N^(-1)sum_{i=1}^N{|delta_{t,i} - delta_{t-1,i}|}, for t=1,...,T
def avg_turn_series(deltas):
     if len(deltas.shape) < 3:
          deltas = np.expand_dims(deltas, axis=2)
     turnover_series = np.abs(deltas[1:, :, 0] - deltas[0:-1, :, 0])
     print(turnover_series.shape)
     return np.mean(turnover_series, 1)


# Show series of turnover. Compute series of turnover if deltas are provided
def show_turn_series(series):
     if series.ndim > 1:
         series = avg_turn_series(series)
     days = [i for i in range(len(series))]
     plt.plot(days, series)
     plt.xlabel("Time")
     plt.ylabel("Turnover")
     plt.legend()
     plt.show()
     return 0


