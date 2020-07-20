# truncate the time decimal to 4 decimals
def truncate(n, decimals=4):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier


# this is the bar method to print the updated bar with the specified params
def barMethod1(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r", time='0'):
    # this is the amount of percent completed
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    # fills the bar to the specified amount
    filledLength = int(length * iteration / total)
    # updates the bar
    bar = fill * filledLength + '-' * (length - filledLength)
    # prints the bar with the correct params
    print(f'\r{prefix} |{bar}| {percent}% {suffix}  >>>  time taken: {truncate(time)} secs', end=printEnd)
    # if the bar is complete, print and stop
    if iteration == total:
        print()
