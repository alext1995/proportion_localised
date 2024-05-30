# Proportion Localised (PL)
Proportion Localised is a novel metric for visual anomaly detection. The metric reports the proportion of anomalies in the dataset sufficiently localized at the best threshold. Note that one image may contain many anomalies.

## Getting started - IPython demo
The Github repo (https://github.com/alext1995/proportion-localised) contains demo data and a notebook. Run through the notebook to get started!

## Advantages of PL
PL works by splitting each ground truth image into individual anomalies (non-overlapping regions), applying the minimum area rotated bounding box to each anomaly, and measuring the IoU between the prediction and each bounding box. If the IoU > 0.3 (the default IoU limit), then the anomaly is classified as localized. A minimum width and height is enforced for each anomaly, and anomalies with a degree of overlap between each other are merged. This metric provides the researcher or professional with an interpretable method of evaluation for anomaly detection algorithms.

P-AUC and AUPRO are the standard metrics used in the field of anomaly detection; however, these have drawbacks in that they output over a small range of values (0.8-1), are not interpretable, are sensitive to noise from ambiguous pixel-wise labels, and in the case of P-AUC, weigh anomalies in proportion to their size.

Proportion Localised mitigates these drawbacks. The rotated bounding boxes ensure that noise from ambiguous pixel labels is mitigated, and the splitting of the ground truth into discrete anomalies means that each anomaly is weighted equally, regardless of its size. As the output is a proportion, the metric reports between 0 and 1, and the metric has a real physical relation.

## Transactions of Machine Learning Research (TMLR) Publication
More details can be found in the Transactions of Machine Learning Research (TMLR) publication: VisionAD, a software package of performant anomaly detection algorithms, and Proportion Localised, an interpretable metric.