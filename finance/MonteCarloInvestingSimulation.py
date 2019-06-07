# Monte Carlo investing simulation
import numpy as np
import matplotlib.pyplot as plt
import pprint


# runs monte carlo simulation to determine proportion of outcomes that lead
# to bins spanning $50,000, e.g., 0-49999, 50000-99999, etc.
# Inputs:
# PV: present value (initial deposit)
# PMT: amount of regular deposit (or withdrawal if negative)
# peryear: number of times deposits/withdrawals are made per year
# t: number of years
# r: average annual ROR as percentage (e.g., use t=6 to represent 6%)
# sd: standard deviation as percentage (e.g., use t=10 to represent 10%)
# N: number of simulations
def simulate(PV, PMT, peryear, t, r, sd, N):
	pp = pprint.PrettyPrinter(indent=4)

	lb = int(-1e6)  # lower bound of bins
	ub = int(1e8)  # upper bound of bins
	inc = 10000  # increment (step size) and bin width
	bins = [[(k,k+inc-1),0] for k in range(lb,ub,inc)]

	# create random array: each row is a full simulation, with data for all years specified
	rand_arr = np.random.normal(r/100.,sd/100.,(N,t*peryear))
	res_arr = []
	for i in range(N):
		res = PV
		# print("---------------"*3)
		# print("amount at month's start: {}".format(res))
		for j in range(t*peryear):
			res = res*(1+rand_arr[i][j]/float(peryear)) + PMT
			rand_arr[i][j] = res
			# print("amount at month's end: {}".format(res))
		res_bin = int(res // inc) - (lb // inc)
		res_arr.append(res)
		bins[res_bin][1] += 1

	percentiles = []
	for i in range(101):
		percentiles.append(np.percentile(res_arr, i))
	print("Based on {} simulations of {} years".format(N, t))
	print("PV:", PV, ",", "Payments:", PMT)
	if percentiles[0] < 0:
		print("*** Warning **********************")
		pct_low = len([i for i in res_arr if i < 0])/len(res_arr)*100
		print("{0:.2f}% of results are negative".format(pct_low))
		print("Decreasing PMT to", PMT+100)
		return simulate(PV, PMT+100, peryear, t, r, sd, N, cumulative=cumulative)
	print("100% chance of ending with more than", percentiles[0])
	print("99% chance of ending with more than", percentiles[1])
	print("95% chance of ending with more than", percentiles[5])
	print("90% chance of ending with more than", percentiles[10])
	print("50% chance of ending with more than", percentiles[50])
	print("25% chance of ending with more than", percentiles[75])
	print("10% chance of ending with more than", percentiles[90])
	print("1% chance of ending with more than", percentiles[99])

	fig, (ax1, ax2) = plt.subplots(2,1,figsize=(16,9))

	ax1.title.set_text("Growth of ${} over {} years with payments of {}".format(PV, t, PMT))
	ax1.text(percentiles[50]-61000, 50, "50%:\n{}".format(int(percentiles[50])))
	ax1.axvline(x=percentiles[50], color='k')  # plot median line
	ax1.text(percentiles[5]-61000, 5, "5%:\n{}".format(int(percentiles[5])))
	ax1.axvline(x=percentiles[5], color='k')  # plot 5th percentile line

	trimbins = trim_bins(bins)
	plt_bins = [b[0][0] for b in trimbins] + [trimbins[-1][0][1]]
	ax1.hist(res_arr, plt_bins)

	ax2.hist(res_arr, plt_bins, cumulative=True, normalized=True)

	plt.draw()
	return trimbins


# removes leading and trailing bins with 0 occurrences
# oldbins is of the form: [[(k,k+49999),0] for k in range(0,9999999,50000)]
def trim_bins(oldbins):
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
		


if __name__ == "__main__":
	pp = pprint.PrettyPrinter(indent=4)

	### Growing stage ### 
	# PV = 20000
	# PMT = 2500
	# peryear = 12
	# years = 14
	# ROR = 7
	# sd = 11.4
	# N = 4000
	# bins = simulate(PV, PMT, peryear, years, ROR, sd, N)


	### Withdraw stage ### with beeb
	PV = 600000
	PMT = -3000
	peryear = 12
	years = 25
	ROR = 7
	sd = 11.4
	N = 4000
	bins = simulate(PV, PMT, peryear, years, ROR, sd, N)


	plt.show()
	