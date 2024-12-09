# aiap19-anthony-tan-beng-wah-095C
AIAP Practical Assessment
------------------------------------------------------------------------------
Name: Tan Beng Wah Anthony
NRIC: Sxxxx095C
Submission Date: 9th Dec 2024
------------------------------------------------------------------------------
I am so sorry I wasn't able to create the run.sh file because of system accessibility issue.
Hence, I have included the Python Notebooks for my 2 temperature prediction models for your reference.
The first model is using Multi Layer Perceptron (MLP) neural network and the second 
model is using Random Forest.

I have also included the following:
- Exploratory Data Analytics (EDA) python notebook
- Requirements.txt file
- Configuration yaml file in .github/workflows folder
- The 2 temperature prediction models in .py script are also included in the src folder
--------------------------------------------------------------------------------
Approaches to the Problem Statement:
1. After retrieving the farm_data from the database and loading them into a dataframe agri, I noticed there were some issues with the data.
2. There were null values in 7 feature columns as well as the "ppm" in some of the rows for Nutrient N, P and K columns.
3. I removed the "ppm" from these 3 columns first and then converted them into float datatype.
4. I replaced all null values with the median value for each column.  After doing this, the data have been cleansed.
5. There was also an issue with Plant Type and Plant Stage data, some were in upper case and some in lower case. As such, I converted all of them to lower case. This is to ensure consistency in the data used in analysis and predictions.
6. Now the cleansed data are ready for EDA.
7. The first analysis is to analyse and compare the environmental and nutrient requirements for all zones ie. Zones A to G. Doing such could help AgroTech Innovations to know the different requirements in each zone.
8. The second analysis is to do similar analysis for Plant Type and Plant Stage.  This analysis compares the requirements at the different stages of growth as well as for the different plant types.
9. The third analysis is to analyse if there is any impact of the previous cycle plant type on the current plant type which can be the same or different.  No substantial impact has been observed.
10. I have included bar charts to compare the temperature, water level and pH requirement for the different plant types as well for the different plant stages.
11. The next steps are to create 2 Machine Learning models to predict the temperature for each data row.  I chose Multi Layer Perceptron (MLP) neural network and Random Forest models.
12. The data cleansing and preparation for these 2 ML models are similar to EDA except that I deliberately concatenated Plant Type and Plant Stage as an appended column and then dropped off the individual Plant Type and Plant Stage columns.  This concatenation is necessary because there is a close correlation between Plant Type and Plant Stage.  
13. On finding the best estimator for MLP hyperparameters, I used for-loop to find out the best number of hidden layers.
14. On finding the best estimator for Random Forest hyperparameters, I used GridSearchCV to find out the best estimators.
15. After using the best hyperparameters, I created the final models and then pipeline for each model.
16. I did a prediction test at the end to check and ensure the final models are producing the close to expected predicted temperature values.
17. MLP neural network is chosen because of its ability to model and solve complex, non-linear relationships between input and output data.
18. Random Forest is chosen because of its robustness and ability to handle a wide variety of data types and tasks while reducing overfitting.
