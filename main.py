import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import data_functions as data_func
import ML as ml

#testing
pred1 = data_func.read_file("C:\SUTD Courses\Year 1 Term3\Modelling Uncertainty\pred1")
pred2 = data_func.read_file("C:\SUTD Courses\Year 1 Term3\Modelling Uncertainty\pred2")
pred3 = data_func.read_file("C:\SUTD Courses\Year 1 Term3\Modelling Uncertainty\pred3")
pred4 = data_func.read_file("C:\SUTD Courses\Year 1 Term3\Modelling Uncertainty\pred4")
FAOSTAT = data_func.read_file("C:\SUTD Courses\Year 1 Term3\Modelling Uncertainty\FAOSTAT")

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
target = data_func.get_target(FAOSTAT, "Value", "O", ["2014-2016", "2015-2017", "2016-2018", "2017-2019", "2018-2020", "2019-2021"])
Victims_of_intentional_homicide = data_func.get_predictor(pred1, "Victims of intentional homicide", "Total", "Total", "Total", "Total", [2014, 2022], "Victims of intentional homicide")
Persons_Convicted = data_func.get_predictor(pred2, "Persons convicted", "Total", "Total", "Total", "Total", [2014, 2022], "Persons Convicted")
Persons_Prosecuted = data_func.get_predictor(pred2, "Persons prosecuted", "Total", "Total", "Total", "Total", [2014, 2022], "Persons Prosecuted")
Burglary = data_func.get_predictor(pred3, "Offences", "by type of offence", "Burglary", "Total", "Total", [2014, 2022], "Burglary")
Theft = data_func.get_predictor(pred3, "Offences", "by type of offence", "Theft", "Total", "Total", [2014, 2022], "Theft")
Corruption = data_func.get_predictor(pred3, "Offences", "by type of offence", "Corruption", "Total", "Total", [2014, 2022], "Corruption")
Fraud = data_func.get_predictor(pred3, "Offences", "by type of offence", "Fraud", "Total", "Total", [2014, 2022], "Fraud")
Serious_assault = data_func.get_predictor(pred4, "Violent offences", "by type of offence", "Serious assault", "Total", "Total", [2014, 2022], "Serious assault")
Robbery = data_func.get_predictor(pred4, "Violent offences", "by type of offence", "Robbery", "Total", "Total", [2014, 2022], "Robbery")
Kidnapping = data_func.get_predictor(pred4, "Violent offences", "by type of offence", "Kidnapping", "Total", "Total", [2014, 2022], "Kidnapping")

# comb = ml.comb_dfs([Victims_of_intentional_homicide, Persons_Prosecuted, Persons_Convicted, target])
# print(comb.iloc[:,2])
learn = ml.learning_main([Victims_of_intentional_homicide, Persons_Prosecuted, Persons_Convicted, target], ["Victims of intentional homicide", "Persons Prosecuted", "Persons Convicted"], ["Value"], iter = 10)
print(learn)