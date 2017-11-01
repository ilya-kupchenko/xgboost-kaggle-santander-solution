## Santander Customer Satisfaction Contest

In this [competition](https://www.kaggle.com/c/santander-customer-satisfaction#description) of 2016 Santander Bank is asking Kagglers to help them identify dissatisfied customers early in their relationship. Our final solution for the contest was implemented in Python, using SkLearn and Pandas libraries.

The data was represented by hundreds of anonymized features to predict if a customer is satisfied or dissatisfied with their banking experience. For each ID in the test data set, you must predict a probability for the TARGET variable. The file should contain a header and have the following format:

ID,TARGET
2,0
5,0
6,0
etc.

### Evaluation

Submissions were evaluated on area under the ROC curve between the predicted probability and the observed target. Our best prediction was 0.824047 and resulted in 2188-th place out of 5183 participants.

The solution is based on the XG-Boost algorithm that was tweaked to gain the best performance.

###Details

Hereafter you can find the detailed PDF-report telling more about the way how our team approached to the final solution.


![Full report](/MLCLASS_Spring16_Kupchenko_Shenoy_1d.pdf)
Format: ![Full report](https://www.dropbox.com/s/m4kvbndkzu7k9bk/MLCLASS_Spring16_Kupchenko_Shenoy_1d.pdf?dl=0)
