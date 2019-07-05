import numpy as np
import matplotlib.pyplot as plt
import pprint

# PV: present value of debt
# t_years: number of years to pay off debt
# peryear: number of payments for year

def pmt_interest(PV, r, t_years, peryear=12):
	n = t_years*peryear
	r = (r/100.)/peryear
	pmt = r*PV/(1-(1+r)**(-n))
	int_loss = n*pmt - PV
	return (pmt, int_loss)

def FV(PMT, PV, r, t_years, peryear=12):
	n = t_years*peryear
	r = (r/100.)/peryear
	fv = PV*(1+r)**n + PMT/r*((1+r)**n - 1)
	return fv

#----------------------------------------------------------------------------------------------------
r_rrsp = 6
PV_locked = 100000
FV_locked = FV(0, PV_locked, r_rrsp, 65-52)
#----------------------------------------------------------------------------------------------------

PV1 = -38000
r = 10
t1 = 5
pmt1, loss1 = pmt_interest(PV1, r, t1)
tot1 = PV1 + loss1

PV_rrsp1 = 24000
t_rrsp1 = (65-52)-t1
FV_rrsp1 = FV(0, PV_rrsp1, r_rrsp, t1)  # growth of initial RRSP amount during debt repayment
FV_rrsp_tot1 = FV(-pmt1, FV_rrsp1, r_rrsp, t_rrsp1)  # RRSP amount at age 65
# FV_rrsp_taxed1 = 0.8*FV_rrsp_tot1

grand_total1 = 0.8*(FV_locked + FV_rrsp_tot1)  # tax-reduced total of RRSPs

print("---"*20)
print("Option 1: Regular payments for {} {}".format(t1, "year" if t1==1 else "years"))
print("Repay debt of ${:,} with interest of {}% over {} {}".format(-PV1, r, t1, "year" if t1==1 else "years"))
print("---"*20)
print("Payments of ${:0,.2f}".format(-pmt1))
print("Total amount paid toward debt: ${:0,.2f}".format(-tot1))
print("Total interest paid: ${:0,.2f}".format(-loss1))
print("---Investment growth---")
print("Initial RRSP amount of ${:,}".format(PV_rrsp1))
print("If you begin investing after {} years with monthly payments of ${:0,.2f}, at 65 you would have ${:0,.2f}".format(t1, -pmt1, FV_rrsp_tot1))
print("At 65, your Locked-in RRSP, which started with ${:,}, grows to ${:0,.2f}".format(PV_locked, FV_locked))
print("In retirement, the RRSP withdrawals are taxed at about 20%, yielding a total RRSP value of ${:0,.2f}".format(grand_total1))
print("Locked-in RRSP ")
print("\n")

#----------------------------------------------------------------------------------------------------

lump = -PV_rrsp1*0.6  # RRSP withdrawal taxed at 40%

PV2 = PV1-lump
t2 = 2.8
pmt2, loss2 = pmt_interest(PV2, r, t2)
tot2 = PV2 + loss2 + lump

PV_rrsp2 = 0
t_rrsp2 = (65-52)-t2
r_rrsp = 6
FV_rrsp2 = FV(0, PV_rrsp2, r_rrsp, t2)  # growth of initial RRSP amount during debt repayment
FV_rrsp_tot2 = FV(-pmt2, FV_rrsp2, r_rrsp, t_rrsp2)

grand_total2 = 0.8*(FV_locked + FV_rrsp_tot2)  # tax-reduced total of RRSPs

print("---"*20)
print("Option 2: Use current RRSP money to pay down debt, followed by {} {} repayment".format(t2, "year" if t2==1 else "years"))
print("Initial lump sum of ${:0,.2f} reduces debt owed to ${:0,.2f}".format(-lump, -PV2))
print("Repay debt of ${:0,.2f} with interest of {}% over {} {}".format(-PV2, r, t2, "year" if t2==1 else "years" ))
print("---"*20)
print("Payments of ${:0,.2f}".format(-pmt2))
print("Total amount paid toward debt: ${:0,.2f}".format(-tot2))
print("Total interest paid: ${:0,.2f}".format(-loss2))
print("---Investment growth---")
print("Initial RRSP amount of ${:0,.2f}".format(PV_rrsp2))
print("If you begin investing after {} years with monthly payments of ${:0,.2f}, at 65 you would have ${:0,.2f}".format(t2, -pmt2, FV_rrsp_tot2))
print("At 65, your Locked-in RRSP, which started with ${:,}, grows to ${:0,.2f}".format(PV_locked, FV_locked))
print("In retirement, the RRSP withdrawals are taxed at about 20%, yielding a total RRSP value of ${:0,.2f}".format(grand_total2))
