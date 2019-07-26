# Monte Carlo investing simulation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import percentileofscore
import pprint
from finlib import median_elem, create_rand_array, recession_adjustment

def trim_bins(oldbins):
    '''
    Removes leading and trailing bins with 0 occurrences from array oldbins.
    oldbins is of the form: [[(k,k+49999),0] for k in range(0,9999999,50000)]
    '''
    bins = oldbins.copy()
    empty = True
    while empty:
        if bins[0][1] == 0:
            bins = bins[1:]
        else:
            empty = False
    empty = True
    while empty:
        if bins[-1][1] == 0:
            bins = bins[:-1]
        if bins[-1][1] > 0:
            empty = False
    return bins


def simulate(PV, PMT, t, r, sd, N=1000, peryear=12):
    '''
    Runs monte carlo simulation to determine possible investing/withdrawing outcomes

    Inputs:
    PV: present value (initial deposit)
    PMT: amount of regular deposit (or withdrawal if negative)
    t: number of years
    r: average annual ROR as percentage (e.g., use t=6 to represent 6%)
    sd: standard deviation as percentage (e.g., use t=10 to represent 10%)
    N: number of simulations
    peryear: number of times deposits/withdrawals are made per year
    '''
    pp = pprint.PrettyPrinter(indent=4)

    lb = int(-1e6)  # lower bound of bins
    ub = int(1e8)  # upper bound of bins
    inc = 50000  # increment (step size) and bin width
    bins = [[(k,k+inc-1),0] for k in range(lb,ub,inc)]

    # create random array: each row is a full simulation, with data for all years specified
    rand_arr = create_rand_array(r, sd, years, N, peryear)
    recession_adjustment(rand_arr, t, r)
    res_arr = []  # stores all final results
    for i in range(N):  # loop through N simulations
        res = PV
        for j in range(t*peryear):  # loop through all periods in a given simulation
            res = res*(1+rand_arr[i][j]) + PMT
            rand_arr[i][j] = res  # update amount at end of period
        res_bin = int(res // inc) - (lb // inc)  # determine index of bin this res belongs to
        res_arr.append(res)
        bins[res_bin][1] += 1  # increase count for the appropriate bin

    res_arr = np.array(res_arr)
    percentiles = np.percentile(res_arr, range(101))  # create array of percentiles to pull from
    # Print summary
    print("Based on {} simulations of {} years".format(N, t))
    print("PV:", PV, ",", "Payments:", PMT)
    if percentiles[0] < 0:
        print("*** Warning **********************")
        pct_low = len([i for i in res_arr if i < 0])/len(res_arr)*100
        print("{0:.2f}% of results are negative".format(pct_low))
        print("PMT changed to", PMT+100)
        PMT += 100
        return simulate(PV, PMT, peryear, t, r, sd, N)
    for p in [0,1,5,10,50,75,90]:
        print("{}% chance of ending with more than ${:,.0f}".format(100-p, int(percentiles[p])))
    print("---"*10)
    print("Probability of getting more than ${:,.0f}: {:,.0f}%".format(PV, 100-percentileofscore(res_arr, PV)))
    print("Probability of getting more than ${:,.0f}: {:,.0f}%".format(1e6, 100-percentileofscore(res_arr, 1e6)))


    # set up graph window
    fig1, (ax1, ax2) = plt.subplots(2,1,figsize=(16,9))

    # fig1, graph 1 -------------------------------------------------
    ax1.title.set_text("Starting with ${:,.0f} over {} years with payments of {:,.0f}".format(PV, t, PMT))
    ax1.text(percentiles[50]-0.05*percentiles[50], 3, "50%:\n{:,}".format(int(percentiles[50])))
    ax1.axvline(x=percentiles[50], color='k')  # plot median line
    ax1.text(percentiles[5]-0.05*percentiles[5], 3, "5%:\n{:,}".format(int(percentiles[5])))
    ax1.axvline(x=percentiles[5], color='k')  # plot 5th percentile line

    trimbins = trim_bins(bins)
    plt_bins = np.array([b[0][0] for b in trimbins] + [trimbins[-1][0][1]])
    counts1, bins1, patches1 = ax1.hist(res_arr, plt_bins)

    # colour the bars
    num_bins = len(plt_bins)
    num_patches = len(patches1)
    for patch, i in zip(patches1, range(1, num_patches+1)):
        sig = 0.9/(1+np.exp(-(i-num_bins)))
        patch.set_facecolor( (0.3, 0.8*(i/num_patches), 0.8*(i/num_patches)) ) # (r,g,b)

    ax1.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))


    # fig1, graph 2 -------------------------------------------------
    num_bins = 25
    step = (plt_bins[-1]-plt_bins[0])/num_bins

    counts2, bins2, patches2 = ax2.hist(res_arr, \
        bins=num_bins, cumulative=-1, density=True)

    ax2.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    # if max(res_arr) > 100000:
    #     plt.xticks(np.arange((min(res_arr)//100000)*100000, (max(res_arr)//100000)*100000, percentiles[98]//500000*50000))

    plt.yticks([i for i in np.arange(0,1.1,0.1)])
    plt.grid(axis='y')

    # colour the bars
    num_patches = len(patches2)
    for patch, i in zip(patches2, range(1, num_patches+1)):
        sig = 0.9/(1+np.exp(-(i-num_bins)))
        patch.set_facecolor( (0.3, 0.8*(i/num_patches), 0.8*(i/num_patches)) ) # (r,g,b)

    # Label the percentiles above each bin
    for i, x in zip(range(num_bins), bins2[:-1]):
        percent = '{:.0f}%'.format(counts2[i]*100)
        # xytext: set location of text to display
        ax2.annotate(percent, xy=(x, 0), xytext=(x+bins2[0]*0.01,counts2[i]))


    # fig2 (timeseries of worst, median, best runs) -----------------

    # find worst, median, best simulations
    ind_worst = np.argmin(res_arr)
    med = median_elem(res_arr)
    ind_med = np.where(np.isclose(res_arr, med))[0][0]
    ind_best = np.argmax(res_arr)
    inds = [ind_worst, ind_med, ind_best]

    # plot timeseries
    fig2, ax3 = plt.subplots()
    ax3.plot(rand_arr[inds].T)

    # return---------------------------------------------------------
    plt.plot()
    return trimbins



if __name__ == "__main__":
    pp = pprint.PrettyPrinter(indent=4)

    ### Deposit stage ### 
    # PV = 30000
    # PMT = 3000
    # years = 12
    # ROR = 7-1.8
    # sd = 11.4
    # bins = simulate(PV, PMT, years, ROR, sd)

    ### Daily habit
    PV = 0
    PMT = 2.30*21
    years = 20
    ROR = 7-1.8
    sd = 11.4
    bins = simulate(PV, PMT, years, ROR, sd)

    ### Withdraw stage ###
    # PMT = -3500
    # years = 25
    # ROR = 7
    # sd = 11.4
    # bins = simulate(PV, PMT, years, ROR, sd)

    plt.show()
    

    # TODO: annotate fig2 to include CAGR beside each timeseries (use ror_with_pmts)