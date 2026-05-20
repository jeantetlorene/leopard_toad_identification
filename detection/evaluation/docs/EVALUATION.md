Fully relying on the test and validation images extracted by the initial, overfitted YOLO model to evaluate performance introduces severe bias, as these subsets may inherently exclude some valid animal instances that the model may have failed to detect. In contrast, conducting manual review of the frames to find all missing organisms is practically intractable. Ensemble inference strategy was deployed to establish a comprehensive ground truth for the held-out locations. Models from the final active learning cycle were utilised to process all unlabelled validation and test footage while maximising the probability of capturing highly cryptic or occluded organisms by using a low confidence threshold. While this highly sensitive configuration successfully captured ambiguous targets, it simultaneously generated approximately 25,000 preliminary predictions per model for validation and test sets each, heavily populated by false-positive environmental artefacts such as shadows and pebbles. To systematically distil this massive output into a manually verifiable subset, cross-model consensus and feature-based diversity selection were applied. Finally, the annotations were thoroughly reviewed and corrected to form the final authoritative YOLO-formatted test and validation dataset. All evaluations are performed by mapping model predictions back to these verified labels using a deterministic image mapping system.

\subsection{Image-Level Evaluation}

The evaluation begins with an assessment of the capacity of the models to function as a filter at the image level. The task was formulated as a binary classification problem to determine the presence or absence of any biological target within a frame, ignoring specific spatial localisation. The manually corrected and verified test set served as the positive ground truth, while the remaining pool of unlabelled images from camera 5Z was assigned a true negative label, representing empty backgrounds. A model predicts a positive image-level label if it generates at least one bounding box with a confidence score exceeding a threshold. Performance at this level was quantified using image-level Recall, mathematically defined as:

\begin{equation}
    \text{Recall} = \frac{TP}{TP + FN}
\end{equation}

where $TP$ represents the number of true positive images correctly identified as containing an animal, and $FN$ represents the number of false negative images containing animals that the model missed. To evaluate the efficiency of the model in reducing the manual human workload by accurately ignoring empty frames, the Specificity was computed using:

\begin{equation}
    \text{Specificity} = \frac{TN}{TN + FP}
\end{equation}

where $TN$ represents the true negative empty frames correctly ignored, and $FP$ denotes the false positive empty frames incorrectly flagged by the model. The Area Under the Receiver Operating Characteristic Curve (ROC-AUC) is also utilised to evaluate binary classification capabilities. The ROC curve plots the True Positive Rate against the False Positive Rate across varying thresholds. The False Positive Rate is mathematically defined as:

\begin{equation}
    \text{FPR} = \frac{FP}{FP + TN}
\end{equation}

However, a critical limitation must be explicitly flagged regarding the use of the ROC-AUC metric for the WLT-UP dataset. Because the continuous time-lapse footage contains a large number of empty true negative frames, the $TN$ denominator in the False Positive Rate equation becomes exceptionally large. 

Consequently, even if a model generates thousands of false positive pebble detections, the False Positive Rate remains suppressed near zero, yielding a misleading ROC-AUC score that completely masks the localisation errors of the model.

To benchmark the fundamental capabilities of the models as an image-level filter, the baseline architectures (YOLO, Faster R-CNN, RT-DETR at Cycle 0, and MegaDetector) were evaluated strictly on the test unlabeled pool. This produced bounded ROC curves explicitly demonstrating their capability to filter empty backgrounds before undergoing active learning.

\subsection{Detection-Level Evaluation}

The second evaluation tier measured the precise localisation and classification accuracy of the predicted bounding boxes within the test subset containing the ground truth annotations. At this instance level, a prediction was considered a True Positive if its IoU with the ground truth bounding box exceeded 0.5, while predictions failing to meet this overlap or misclassifying the class were deemed False Positives. To assess the exactness of the positive detections, Precision was calculated as:

\begin{equation}
    \text{Precision} = \frac{TP}{TP + FP}
\end{equation}

F1-score was computed to provide a single harmonic mean that balances both precision and recall at a specific threshold:


\begin{equation}
    \text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
\end{equation} 

The detection-level evaluation prioritised threshold-independent metrics derived from the Precision-Recall (PR) curve. The Average Precision (AP) for each distinct taxonomic class was calculated as the exact area under its respective PR curve:


\begin{equation}
    \text{AP} = \int_{0}^{1} P(R) \, dR
\end{equation}

where $P(R)$ represents precision as a function of recall. A unified performance score for the WLT, other amphibian, and small mammal classes was achieved by computing the mean Average Precision (mAP), which averages the Average Precision values across all $N$ classes.
\begin{equation}
    \text{mAP} = \frac{1}{N} \sum_{i=1}^{N} \text{AP}_i
\end{equation}

\subsection{Threshold Sweeping and Class-Specific Calibration}
A significant challenge in evaluating the models is that the metrics are highly dependent on a selected confidence threshold, which introduces evaluation bias. To assess the absolute capacity of the models to capture all possible scenarios of the target organisms, threshold sweeping was executed across the annotated subsets. This procedure involved sorting all predictions in descending order based on their confidence scores and iteratively calculating bounding-box precision and recall (using an Intersection over Union criterion of 0.5) at every continuous confidence level to construct the PR curves. Because the three classes exhibit varying levels of visual distinctiveness and occurrence frequencies, applying a uniform threshold across the entire network architecture is highly suboptimal. A single cutoff value might severely reduce recall for a target species while simultaneously generating an overwhelming volume of false positives for another class. By analysing the swept PR curves strictly on the annotated validation set, class-specific optimal thresholds were calibrated by identifying the highest confidence level that strictly maximizes Recall while minimizing False Positives (i.e., maximizing Precision for that specific recall ceiling). To objectively quantify the deployment benefits of this dynamic calibration, the models were subsequently evaluated on the independent, annotated test set. For each taxonomic class, the detection performance achieved using the optimized threshold was directly contrasted against the baseline performance generated by a generic 0.5 confidence threshold, demonstrating the necessity of targeted threshold tuning for high-recall wildlife monitoring.


\subsection{Effect of Preprocessing} 
% The baseline is the object detection performance evaluated on the raw, unenhanced camera trap images prior to the application of the Contrast Limited Adaptive Histogram Equalisation technique. For the effect of preprocessing, the exact comparison pits the Cycle zero architectures trained on raw, unenhanced camera trap images directly against their identical structural counterparts trained on images enhanced using Contrast Limited Adaptive Histogram Equalisation. Contrast Limited Adaptive Histogram Equalisation is defined as a localised enhancement algorithm that improves feature visibility by redistributing pixel intensities in degraded night-time imagery. You will evaluate the YOLO, RT-DETR, and Faster R-CNN models under both data conditions simultaneously. The threshold-swept metrics, specifically mean Average Precision and mean Average Recall, extracted from the unenhanced models, serve as the definitive baseline against which the enhanced models are measured. This isolates whether the mathematical manipulation of pixel contrast universally improves deep feature extraction before any dynamic data sampling is introduced. To evaluate the effect of preprocessing, the baseline models consist of the chosen object detection architectures trained and evaluated exclusively at Cycle zero using raw, unenhanced nighttime infrared images.

% Here is the outline of the specific reporting elements required for the preprocessing subsection:

% Comparative Preprocessing Performance Table: This table must directly contrast the threshold-independent metrics of the baseline architectures—specifically the YOLO, RT-DETR, and Faster R-CNN models—trained on raw, unenhanced camera trap images against their exact structural counterparts trained on the CLAHE-enhanced dataset. The table should explicitly report the mean Average Precision, alongside the mean Average Recall.

% Overlapping Precision-Recall Curves: The results must include a graphical representation plotting precision against recall at varying classification thresholds for both the unenhanced and enhanced models.


\subsection{Effect of Architecture} 
% The baseline consists of the standard configurations of the YOLO, RT-DETR, and Faster R-CNN models, initialised solely with general-domain weights, serving as the benchmark before any specialised transfer learning or active learning optimisations are introduced. This comparative analysis establishes the foundational benchmark regarding detection accuracy, inference speed, and parameter efficiency before any domain-specific active learning optimisations are introduced into the computational pipeline. You will compare these three models against one another exclusively at Cycle zero, utilizing the exact same preprocessed data and domain-specific weight initializations. By holding the training data distribution and the transfer learning initializations entirely constant, the resulting discrepancies in mean Average Precision, inference speed, and parameter count directly reflect the inherent spatial localization and classification capabilities of each specific network design. To evaluate the effect of architecture, the foundational baselines comprise three distinct structural frameworks evaluated strictly at Cycle zero: the single-stage You Only Look Once network, the Real-Time Detection Transformer, and the two-stage Faster Region-based Convolutional Neural Network equipped with a ResNet50 feature extraction backbone.

% Comprehensive Architectural Benchmarking Table: You must construct a detailed table reporting the quantitative performance of the YOLO, RT-DETR, and Faster R-CNN architectures evaluated at Cycle zero. The columns must include threshold-independent accuracy metrics, specifically the mean Average Precision at an Intersection over Union of 0.5 (mAP50) and averaged across 0.5 to 0.95 (mAP50-95), alongside mean Average Recall. Crucially, to assess practical deployability, the table must also report computational metrics including Model Size (measured in millions of parameters), Inference Speed (measured in milliseconds per image or Frames Per Second), and computational complexity, which is measured in Giga Floating Point Operations Per Second (GFLOPs).

% Architecture-Specific Confusion Matrices: To transparently reveal the specific classification vulnerabilities of each network paradigm, generate individual confusion matrices for YOLO, RT-DETR, and Faster R-CNN at the baseline stage. A confusion matrix is a tabular representation that cross-references predicted categories against actual ground-truth labels to identify true positives, false positives, and false negatives. This will explicitly demonstrate whether a specific architecture struggles disproportionately with background noise, or if it exhibits high inter-class confusion between the Western Leopard Toad and other amphibians.


\subsection{Effect of Transfer Learning}
% The baseline comprises the scratch models, which bypass the intermediate domain-specific pretraining phase and are directly fine-tuned on the target dataset using only their original general-domain initialisations. For the effect of transfer learning, the comparative baseline is established by models trained entirely from scratch, utilising standard random weight initialisations without prior domain exposure. You will compare these scratch-trained baseline models directly against identically configured architectures that have been initialised with weights pre-trained on massive, domain-specific datasets. This specific comparison, also conducted at Cycle zero using the preprocessed dataset, mathematically isolates the benefits of domain adaptation. It demonstrates exactly how prior structural knowledge mitigates catastrophic forgetting, improves generalisation to unseen backgrounds, and accelerates mathematical convergence compared to networks forced to learn visual features from nothing. To evaluate the effect of transfer learning, the baseline models are explicitly defined as standard network configurations, evaluated at Cycle zero, that are initialised with original weights pretrained exclusively on large-scale, general-domain datasets.

% Here is the detailed outline of the specific tables and figures required to rigorously report the effect of transfer learning in your manuscript, structured as requested:

% *   **Comprehensive Transfer Learning Performance Table:** You must include a table directly comparing the baseline architectures trained entirely from scratch against their identical structural counterparts initialized with domain-specific pre-trained weights at Cycle zero. The columns must clearly report the exact network architectures (YOLO, RT-DETR, Faster R-CNN) alongside threshold-independent evaluation metrics, specifically mean Average Precision (mAP50 and mAP50-95), mean Average Recall (mAR), and the total number of trainable parameters.

% *   **Comparative Precision-Recall Curves:** To demonstrate the specific impact on detection quality across varying confidence thresholds, include overlapping Precision-Recall (PR) curves for the scratch and transfer-learned models evaluated on the validation set. By plotting Precision against Recall, you graphically represent how the domain-specific pretraining expands the total area under the curve (Average Precision). This figure is critical for justifying that transfer learning inherently improves the model's capacity to maintain higher precision at the deliberately low confidence thresholds required for high-recall wildlife monitoring. 


\subsection{Effect of Active Learning}
% For the effect of active learning, the exact comparison evaluates the targeted dynamic sampling strategy against an unguided, static baseline. You will compare the progression of the model optimised through your specific difficulty-calibrated uncertainty and diversity sampling pipeline against the progression of an identical model trained using a purely random sampling strategy. This comparison must span the entire training progression from Cycle one through the final iteration. By plotting the mean Average Precision and class-specific recall of both strategies across successive cycles, you isolate the exact efficiency gained by algorithmically querying highly informative, complex boundary cases rather than simply accumulating arbitrary unlabelled samples.

% Active Learning Progression Table: A comprehensive table documenting the cycle-by-cycle evolution of the detection models. Required columns include Cycle Number, Cumulative Labelled Images (or annotation budget), mAP50, mAP50-95, Precision, and Recall. This table provides the fundamental mathematical proof that the targeted querying mechanism steadily enhanced spatial localisation and classification accuracy as more complex boundary cases were absorbed into the training pool.

% Class-Specific Performance Trajectory Plot: A line plot tracking the individual recall and mAP progression for the three distinct taxonomic categories (Western Leopard Toad, other amphibians, small mammals) across successive training cycles. This figure is critical for demonstrating that your active learning pipeline successfully recognised the difficulty of the highly camouflaged, endangered target species and adaptively queried it to force performance improvements over the more common background classes.

% Overlapping Confidence Score Distribution Plot: A histogram or Kernel Density Estimation plot evaluated at the final active learning cycle, mapping the density of true positive predictions against false positive artefacts. Placed at the culmination of the active learning results, this plot visually justifies the extraction of the deliberately low, class-specific operational threshold required to guarantee near-perfect recall for the Western Leopard Toad during real-world deployment.