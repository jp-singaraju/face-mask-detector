# truncate the time decimal to 4 decimals
def truncate(n, decimals=4):
    # get the multiplier value
    multiplier = 10 ** decimals

    # return the truncated time in specified decimals
    return int(n * multiplier) / multiplier


# this is the bar method to print the updated progress bar with the specified params
def bar_method(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', print_end="\r", time='0'):
    # this is the calculation for the amount of percent completed
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))

    # fills the progress bar to the specified amount (filled_length)
    filled_length = int(length * iteration / total)

    # updates the progress bar
    bar = fill * filled_length + '-' * (length - filled_length)

    # prints the progress bar with the correct params
    print(f'\r{prefix} |{bar}| {percent}% {suffix}  >>>  time taken: {truncate(time)} secs', end=print_end)

    # if the bar is complete, print and stop
    if iteration == total:
        print()
