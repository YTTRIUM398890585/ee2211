import pandas as pd
import matplotlib.pyplot as plt


dataframe = pd.read_csv('AnnualMotorVehiclePopulationbyVehicleType.csv')

# privateBusIndexList = dataframe[dataframe['type'] == "Private buses"].index.tolist()
# print(privateBusIndexList)

privateBusIndexList = dataframe.loc[dataframe['type'] == "Private buses"].index
excursionBusIndexList = dataframe.loc[dataframe['type'] == "Excursion buses"].index
omniBusIndexList = dataframe.loc[dataframe['type'] == "Omnibuses"].index

privateBusDataframe = dataframe.loc[privateBusIndexList]    
excursionBusDataframe = dataframe.loc[excursionBusIndexList]    
omniBusDataframe = dataframe.loc[omniBusIndexList]    

# or without extracting the index first
# privateBusDataframe = dataframe.loc[dataframe['type'] == "Private buses"]
# excursionBusDataframe = dataframe.loc[dataframe['type'] == "Excursion buses"]
# omniBusDataframe = dataframe.loc[dataframe['type'] == "Omnibuses"]

plt.plot(privateBusDataframe['year'], privateBusDataframe['number'], color='green')
plt.plot(excursionBusDataframe['year'], excursionBusDataframe['number'], color='orange')
plt.plot(omniBusDataframe['year'], omniBusDataframe['number'], color='blue')

plt.show()

yearCut = 2017

print("cutting away years after {}".format(yearCut))
privateBusDataframe = privateBusDataframe.loc[privateBusDataframe['year'] <= yearCut]
excursionBusDataframe = excursionBusDataframe.loc[excursionBusDataframe['year'] <= yearCut]
omniBusDataframe = omniBusDataframe.loc[omniBusDataframe['year'] <= yearCut]

plt.plot(privateBusDataframe['year'], privateBusDataframe['number'], color='green')
plt.plot(excursionBusDataframe['year'], excursionBusDataframe['number'], color='orange')
plt.plot(omniBusDataframe['year'], omniBusDataframe['number'], color='blue')

plt.show()
# privateBusNumber = []
# privateBusYear = []

# excursionBusNumber = []
# excursionBusYear = []

# omniBusNumber = []
# omniBusYear = []


# for i in range(0, len(dataframe['year'])):
#     if dataframe['type'][i] == "Private buses":
#         privateBusNumber.append(dataframe['number'][i])
#         privateBusYear.append(dataframe['year'][i])
#     elif dataframe['type'][i] == "Excursion buses":
#         excursionBusNumber.append(dataframe['number'][i])
#         excursionBusYear.append(dataframe['year'][i])
#     elif dataframe['type'][i] == "Omnibuses":
#         omniBusNumber.append(dataframe['number'][i])
#         omniBusYear.append(dataframe['year'][i])


# plt.plot(privateBusYear, privateBusNumber, color='green')
# plt.plot(excursionBusYear, excursionBusNumber, color='orange')
# plt.plot(omniBusYear, omniBusNumber, color='blue')

# plt.show()

# print("done")