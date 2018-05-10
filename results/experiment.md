Experiment and Results

The network was presented with a dataset of 60,000 observations and 128 features, with a 10-class label evenly balanced between all ten classes. This dataset was used to test the performance of the model.

In order to ensure the results were as unbiased as possible, the dataset was randomly shuffled and split into 3 different buckets. 40,000 shuffled rows became the training set, another 10,000 became a testing/monitoring set, and a final 10,000 were used for a validation set. This ensured that the model had no chance of overfitting to the training set, as well as ensuring honesty in the results by not hyper-parameter tuning to the test set.

The model was trained via stochastic gradient descent with mini-batching, for 200,000 iterations or 5 Epochs of the training data. This took approximately 2 hours on a 2015 Macbook Pro. Batch size was 200, and training rate was 1e-3. The results of the model are discussed below:

3.1 Accuracy

The model accuracy was high for a ten-class problem at 74.29% on the validation set. Naive guess would give a performance of 10%, so the model does 7.4 times better than random. The accuracy on the test set was XX.XX percent, and the accuracy on the training data was XX.XX percent. This indicates that the model was generalising well and that overfitting was not an issue.

3.2 Extensive Analysis

3.2.1 Confusion Matrix

The confusion matrix is a powerful visualisation of model performance. The confusion matrix of the network's performance on the validation data suggests that the model struggles to recognise classes 6 and 4 - there are more incorrect predictions of data in these classes than correct ones. Most other classes, however, are predicted well and this reflects the high accuracy scores explored in the previous section.

3.2.2 Precision, Recall and F-score

Precision, Recall and F-score are analysed as a set, because they all easily computed from the confusion matrix. All these metrics are designed for binary classification, but they can be generalised to multiclass problems via one-vs-all scoring and averaging by class fequency. The results achieved by the model are listed below:

* Precision: 0.754
* Recall: 0.743
* F-score: 0.723

The scores are all similar, but it is surprising that F-score is lower than Precision and Recall, given that it is by definition the harmonic mean of the two. This occurred through the aggregation process, if the F-score were calculated off the aggregated precision and recall scores it would be 0.748

3.2.3 ROC and ROC-AUC

The ROC-AUC score is more forgiving than Precision, Recall and F-score, since it deals with probabilities rather than predictions. Specifically, incorrect predictions where the true class had a reasonable probability are taken into account. The ROC-AUC score is "excellent" at 0.969, and the individual ROC curves are provided below:


3.2.4 Training Time and Convergence

The model saw rapid improvement very quickly but further gains were much slower and progress was not monotonic. Graphing this over training iterations, it is clear that the model was able to push past local minima towards a better solution. A deep trough early in training can be explained by the momentum of parameter updates carrying the model past a promising initial solution, since the rate of parameter change was initially large. Also, while the rate of improvement had diminished, the visualisation suggests that better results could be achieved given additional iterations. This was not explored in the interests of keeping runtimes feasible.




