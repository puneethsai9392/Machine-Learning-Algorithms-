def leap(year):
    if year%400==0 and (year%4==0 or year%100!=0):
        print('leap')
    else:
        print('Not leap')
year=int(input('enter the year'))
print(leap(year))
    