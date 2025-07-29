import matplotlib.pyplot as plt

def plot_price(price_series, title="Price"):
    plt.figure()
    plt.plot(price_series)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.grid(True)
    plt.show() 