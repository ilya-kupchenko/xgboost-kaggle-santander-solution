## Santander Customer Satisfaction Contest

In this 2016 [competition](https://www.kaggle.com/c/santander-customer-satisfaction#description), Santander Bank is asking Kagglers to help them identify dissatisfied customers early in their relationship. Our final solution for the contest was implemented in Python, using SkLearn and Pandas libraries.

The data was represented by hundreds of anonymized features to predict if a customer is satisfied or dissatisfied with their banking experience. For each ID in the test data set, we predicted a probability for the TARGET variable. The output file needed to contain a header and have the following format:

ID|TARGET
---------
2|0
---
5|0
---
6|0
---
etc.
----

### Evaluation

Submissions were evaluated on the area under the ROC curve between the predicted probability and the observed target. Our best prediction was 0.824047 and resulted in 2188-th place out of 5183 participants.

The solution is based on the XG-Boost algorithm that was tweaked to gain the best performance.

### Full version of the report

Below you can find the detailed PDF-report, telling more in detail how our team approached generating our solution.


![Full report](/MLCLASS_Spring16_Kupchenko_Shenoy_1d.pdf)
