# Define constants
PV_debt = -38000
PV_rrsp = 24000  # liquid RRSPs
PV_locked = 100000
cur_age = 52
retirement_age = 65


def pmt_interest(PV, r, t_years, peryear=12):
    '''
    PV: present value of debt
    r: annual rate of return percentage (e.g, r=7 for 7%)
    t_years: number of years to pay off debt
    peryear: number of payments per year
    
    Returns tuple where first element is the required payment 'peryear'
    times per year, and second element is the total interest paid.
    '''
    n = t_years*peryear
    r = (r/100.)/peryear
    pmt = r*PV/(1-(1+r)**(-n))
    int_loss = n*pmt - PV
    return (pmt, int_loss)

def FV(PMT, PV, r, t_years, peryear=12):
    '''
    PMT: amount of each payment, made 'peryear' times per year
    PV: present value
    r: annual rate of return percentage (e.g, r=7 for 7%)
    t_years: number of years
    peryear: number of payments per year
    
    Returns the future value of the annuity, compounded 'peryear' times
    per year over t_years.
    '''
    n = t_years*peryear
    r = (r/100.)/peryear
    fv = PV*(1+r)**n + PMT/r*((1+r)**n - 1)
    return fv


def simulate_retirement(PV_debt, r_debt, t_debt, PV_rrsp, r_rrsp, lump, PV_locked, cur_age, retire_age, option_num=1):
    '''
    PV_debt: present value of debt (negative)
    r_debt: annual percentage of debt interest (e.g, r_debt=10 for 10%)
    t_debt: number of years to pay off debt
    PV_rrsp: current amount held in liquid RRSPs (positive)
    r_rrsp: annual rate of return on RRSP investments
    lump: amount of RRSP withdrawals used to repay debt initially
    PV_locked: current amount held in Locked-in RRSPs
    cur_age: age, in years, at present
    retirement_age: age, in years, at retirement
    option_num: number to print before the generated report
    
    Prints a report outlining debt repayment and investment growth, as well as
    an overall total investment value at retirement.
    '''
    marginal_tax_rate = 0.40
    retirement_tax_rate = 0.2
    lump = lump*(1 - marginal_tax_rate)
    PV = PV_debt - lump
    pmt, loss = pmt_interest(PV, r_debt, t_debt)
    tot = PV_debt + loss

    PV_rrsp_reduced = PV_rrsp + lump/(1 - marginal_tax_rate)
    t_rrsp = (retire_age-cur_age)-t_debt
    FV_rrsp = FV(0, PV_rrsp_reduced, r_rrsp, t_debt)  # growth of initial RRSP amount during debt repayment
    FV_rrsp_tot = FV(-pmt, FV_rrsp, r_rrsp, t_rrsp)

    # Locked-in RRSP growth
    FV_locked = FV(0, PV_locked, r_rrsp, retirement_age-cur_age)

    grand_total = (1 - retirement_tax_rate)*(FV_locked + FV_rrsp_tot)  # tax-reduced total of RRSPs

    year_str = "year" if t_debt==1 else "years"
    print("---"*20)
    print("Option", option_num)
    print("Initial lump sum of ${:,} is taxed at about 40%, yielding ${:,}, which reduces debt owed to ${:0,.2f}".format(int(-lump/0.6) , int(-lump), int(-PV)))
    print("Repay debt of ${:,} with interest of {}% over {} {}".format(int(-PV), r_debt, t_debt, year_str))
    print("---"*20)
    print("Payments of ${:0,.2f}".format(-pmt))
    print("Total amount paid toward debt: ${:0,.2f}".format(-tot))
    print("Total interest paid: ${:0,.2f}".format(-loss))
    print("---Investment growth---")
    print("Initial RRSP amount of ${:0,.2f}".format(PV_rrsp_reduced))
    print("If you begin investing after {} years with monthly payments of ${:0,.2f}, at {} you would have ${:0,.2f}".format(t_debt, -pmt, retire_age, FV_rrsp_tot))
    print("At {}, your Locked-in RRSP, which started with ${:,}, will have grown to ${:0,.2f}".format(retire_age, PV_locked, FV_locked))
    print("RRSP withdrawals in retirement should be taxed at about 20%, yielding a total net RRSP value of ${:0,.2f}".format(grand_total))
    print("\n")


# Option 1: repay as usual
simulate_retirement(
    PV_debt=PV_debt,
    r_debt=10,
    t_debt=5,
    PV_rrsp=PV_rrsp,
    r_rrsp=5,
    lump=0,
    PV_locked=100000,
    cur_age=52,
    retire_age=65,
    option_num=1)

# Option 2: withdraw all RRSP to put toward debt
simulate_retirement(
    PV_debt=PV_debt,
    r_debt=8,
    t_debt=2.75,
    PV_rrsp=PV_rrsp,
    r_rrsp=5,
    lump=-PV_rrsp,
    PV_locked=100000,
    cur_age=52,
    retire_age=65,
    option_num=2)

# Option 3: withdraw half of RRSP to put toward debt
simulate_retirement(
    PV_debt=PV_debt,
    r_debt=8.9,
    t_debt=3.75,
    PV_rrsp=24000,
    r_rrsp=5,
    lump=-PV_rrsp/2,
    PV_locked=100000,
    cur_age=52,
    retire_age=65,
    option_num=3)

# TODO: what if pmts of 800 per month with 0 starting debt?
# TODO: add multiple sources of debt with different interest rates (array of tuples)