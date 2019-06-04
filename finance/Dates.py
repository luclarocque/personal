# Date module
import datetime

# curdate is a datetime.date object, years and months are the number of years and
#  months to go back, respectively.
def date_back(curdate, years, months):
	extra_years = months // 12
	extra_months = months % 12
	m = (curdate.month - extra_months) % 12
	if m == 0:
		m = 12
	if m > curdate.month:
		extra_years += 1
	y = curdate.year - years - extra_years
	d = curdate.day
	if m == 2 and d > 28:
		d = 28
	elif m in [4,6,9,11] and d > 30:
		d = 30
	return datetime.date(y,m,d)