# Shaolin AI 19 - Model Optimization - Model Audit
This exercise looks at model optimization by auditing a model made by a hypothetical colleague "Steve" to analyze potential model bias. 
"Steve" created a model based on COMPAS data of Broward County, FL prisoners to predict whether or not they were likely to reoffend. He was then instruced to analyze whether or not the model showed racial bias, and to correct for that bias.

After performing this mock audit, I repeated the exercise myself starting from the same base data.

## Model Audit
Steve provided minimal explanation and commentary on his code, it seemed to me that he had limited grasp of what he was doing. These are my notes on his process:
#### Data Cleaning
* Several columns he kept were near-duplicates or closely related data, potentially leading to over-fitting of the data on those features.
* In particular, multiple columns ('is_recid', 'is_violent_recid', 'two_year_recid') record whether the prisoner became a recidivist under different metrics. Choosing one of these as the model target while keeping the other two in the model, as Steve did, is likely to inflate accuracy of the model by giving it in effect a "cheat sheet."
* Inclusion of the COMPAS score itself within the model conflicts with the stated goal to "Study if your classifiers are more or less fair than the COMPAS classifier." Once again there are several versions of this score included.
* Steve checked for nulls in his chosen columns, saw there were none, and then dropped nulls in the next code cell. This indicated to me he did not understand what the meaning of these steps are.
#### Model construction
* Steve chose 'two_year_recid' as his target column, while this is a valid choice I think 'is_recid' would be a better choice for analysis of whether the prisoners became recidivsts without including a specific timeframe.
* Two columns had binary categorical data,'sex' and 'c_charge_degree'. Dummy encoding this columns created two sets of mirror-image features in the model, again potentially leading to overfitting on these features. One of the output columns from each of these could have been dropped.
* Steve chose a Support Vector Classification model "because I like them."  This again shows a lack of understanding of why a particular model should be chosen for a particular task. SVC's also make analysis of how individual features contribute to the decision function difficult, making them not well suited for this exercise.
* Steve did not scale his features before fitting the model, which with a different model type would have made interperetation of feature contribution to the model easier.
#### Model Analysis
* Steve calculated accuracy and made a confusion matrix, and his full analysis was "This is a very good and fair model because it is very accurate and predicts very well." While technically correct, he did not provide any analysis of the confusion matrix or provide precision, recall or F1 scores once again leading me to think he does not understand the meaning of these metrics.
* He did not analyze "fairness" at all, let alone take measures to mitigate potential bias in his model

## Analyzing and Reducing Bias in Audited Model
Starting from the model produced above, I took several steps:
#### Feature Analysis
The only method I found to extract a feature analysis from this SVC model was permutation_importance from sklearn, this was a computationally expensive step and took several minutes each time running the code. Features were assigned an importance score and sorted by magnitude of absolute value:
```
is_recid: 0.45221729490022167
age: 0.002383592017738345
priors_count: 0.0010532150776052852
is_violent_recid: -5.543237250554833e-05
v_decile_score: -5.543237250554833e-05
sex_Female: 0.0
juv_fel_count: 0.0
decile_score: 0.0
juv_misd_count: 0.0
juv_other_count: 0.0
decile_score.1: 0.0
c_charge_degree_M: 0.0
c_charge_degree_F: 0.0
age_cat_25 - 45: 0.0
age_cat_Greater than 45: 0.0
age_cat_Less than 25: 0.0
race_African-American: 0.0
race_Asian: 0.0
race_Caucasian: 0.0
race_Hispanic: 0.0
race_Native American: 0.0
race_Other: 0.0
sex_Male: 0.0
```
As I expected, the most important features included the features closely related to the target (is_recid, is_violent_recid) and one measure of the COMPAS model output (v_decile_score). While no importance was assigned to any racial categories, a model not using the erroneous features may have shown bias.
#### Model performance on selected features
I extracted actual and predicted values for African American and Caucasian prisoners and constructed confucion matrices and model scores for these subsets of data:
```
Metrics of model fit for data on Caucasians
[[354  15]
 [  1 231]]
Accuracy: 0.9734
Precision: 0.9390
Recall: 0.9957
F1 Score: 0.9665


Metrics of model fit for data on African-Americans
[[425  47]
 [  0 473]]
Accuracy: 0.9503
Precision: 0.9096
Recall: 1.0000
F1 Score: 0.9527
```
While the model scored very highly for both groups, I did notice that precision is slightly higher for Caucasians than African Americans, while recall is slightly higher for African Americans than for Caucasians. This indicates that the model has a lower proportion of false positives \( predicted recidivsm that did not occur\)for Caucasians, and a lower proportion of false negatives \( actual recidivism that was not predicted\)for African-Americans.
Rephrased more directly, the model can be seen as biased to predicting recidivism among Caucasians. However, this is in a limited sense, and given the high and very close metrics across these groups I do not see it as being meaningful.

## Constructing a New Model
#### Changes Made
From my analysis of Steve's model, I had several ideas for improvement:
* Set 'is_recid' as target, drop 'two_year_recid' and 'is_violent_recid'
* Drop 'decile_score', 'decile_score.1', 'v_decile_score' to not have COMPAS score itself affect model
* Drop age_cat, after dummy encoding drop 'c_charge_degree_F', 'sex_Female'
* Scale data with StandardScaler
* Use Logistic Regression
These were implemented and rerun on the source data.
I then performed the same analysis as above:
#### Overall Model Feature Importance and Metrics
```
sex_Male: 0.352
juv_fel_count: 0.238
race_Native American: 0.219
race_African-American: 0.205
c_charge_degree_M: -0.200
priors_count: 0.162
juv_other_count: 0.145
race_Hispanic: -0.109
race_Caucasian: 0.077
race_Other: -0.045
age: -0.042
race_Asian: 0.036
juv_misd_count: 0.024
```
This model does indeed apply more importance to some racial categories than others.

```
Model metrics
[[479 428]
 [473 424]]
Accuracy: 0.5006
Precision: 0.4977
Recall: 0.4727
F1 Score: 0.4848
```
Overall this model scores poorly.

#### Model Metrics of Specific Features
```
Metrics of model fit for data on Caucasians
[[183 152]
 [138 135]]
Accuracy: 0.5230
Precision: 0.4704
Recall: 0.4945
F1 Score: 0.4821


Metrics of model fit for data on African-Americans
[[218 206]
 [286 249]]
Accuracy: 0.4870
Precision: 0.5473
Recall: 0.4654
F1 Score: 0.5030
```
It at least scores equally poorly for Caucasians and African Americans, showing little bias.