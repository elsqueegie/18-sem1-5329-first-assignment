Experiment and Results

The network was presented with a dataset of 60,000 observations and 128 features, with a 10-class label evenly balanced between all ten classes. This dataset was used to test the performance of the model.

In order to ensure the results were as unbiased as possible, the dataset was randomly shuffled and split into 3 different buckets. 40,000 shuffled rows became the training set, another 10,000 became a testing/monitoring set, and a final 10,000 were used for a validation set. This ensured that the model had no chance of overfitting to the training set, as well as ensuring honesty in the results by not hyper-parameter tuning to the test set.

The model was trained via stochastic gradient descent with mini-batching, for 200,000 iterations or 5 Epochs of the training data. This took approximately 2 hours on a 2015 Macbook Pro. Batch size was 200, and training rate was 1e-3. The results of the model are discussed below:

3.1 Accuracy

The model accuracy was high for a ten-class problem at 74.29% on the validation set. Naive guess would give a performance of 10%, so the model does 7.4 times better than random. The accuracy on the test set was XX.XX percent, and the accuracy on the training data was XX.XX percent. This indicates that the model was generalising well and that overfitting was not an issue.

3.2 Extensive Analysis

3.2.1 Confusion Matrix

The Confusion Matrix is a powerful visualisation of model performance. The Confusion Matrix of the network's performance on the validation data suggests that the model struggles to recognise classes 6 and 4 - there are more incorrect predictions of data in these classes than correct ones. Most other classes, however, have high accuracy as explored in the previous section.

3.2.1 Precision, Recall and F-score

Precision, Recall and F-score are analysed as a set, because they all 

The model performs similarly well across other metrics. Precision, recall, f-score and ROC-AUC score are all designed for 2-class problems, but can be restructured for multiclass problems using a 1-vs-all approach and averaging over the classes. The following results were obtained through this process: 

* Precision: 0.754
* Recall: 0.743
* F-score: 0.723
* ROC-AUC: 0.969

Of these metrics, ROC-AUC is the most impressive, because it is the most forgiving. Since ROC-AUC is scored on probabilities rather than predictions, it does not penalise incorrect predictions to the same degree as the other metrics, provided the probability is still relatively high. The disparity between ROC-AUC score and the other scores indicate that there are many edge-cases in the predictions.

Again, ROC-AUC, Precision, Recall and F-score are all similar for train, test set, set and validation set. This demonstrates that the rules learned by the model generalise well to other data.



