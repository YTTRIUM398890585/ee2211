import pandas as pd
import matplotlib.pyplot as plt

dataframe = pd.read_csv('GovernmentExpenditureonEducation.csv')

expenditureList =  dataframe['total_expenditure_on_education'].tolist()
yearList = dataframe['year'].tolist()

plt.plot(yearList, expenditureList, color='blue', marker='x')
plt.xlabel('Year')
plt.ylabel('Total Expenditure on Education')
plt.title('Government Expenditure on Education')
plt.legend({'Total Expenditure on Education'})
plt.show()

print("done")

