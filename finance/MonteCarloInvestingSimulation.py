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
def simulate(PV, PMT, peryear, t, r, sd, N, cumulative=False):
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

	print("PV:", PV, ",", "Payments:", PMT)
	print("Based on {} simulations of {} years".format(N, t))
	print("50% chance of ending with more than", np.median(res_arr))
	print("90% chance of ending with more than", np.percentile(res_arr, 10))
	print("95% chance of ending with more than", np.percentile(res_arr, 5))
	print("99% chance of ending with more than", np.percentile(res_arr, 1))
	print("100% chance of ending with more than", np.min(res_arr))

	trimbins = trim_bins(bins)
	plt_bins = [b[0][0] for b in trimbins] + [trimbins[-1][0][1]]
	plt.hist(res_arr, plt_bins, cumulative=None)
	# plt.axis([plt_bins[0],plt_bins[-1]+inc,0, max([b[1] for b in trimbins])])
	plt.text(1,1,"PV={}".format(PV))
	plt.show()

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
	PV = 20000
	PMT = 2000
	peryear = 12
	years = 14
	ROR = 7
	sd = 11.4
	N = 10000
	bins = simulate(PV, PMT, peryear, years, ROR, sd, N)


	### Withdraw stage ### with beeb
	# PV = 600000
	# PMT = -2800
	# peryear = 12
	# years = 25
	# ROR = 7
	# sd = 11.4
	# N = 10000
	# bins = simulate(PV, PMT, peryear, years, ROR, sd, N, cumulative=True)


	