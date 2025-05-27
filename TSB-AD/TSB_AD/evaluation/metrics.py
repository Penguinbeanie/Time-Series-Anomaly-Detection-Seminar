from .basic_metrics import basic_metricor, generate_curve
import numpy as np

def get_metrics(score, labels, slidingWindow=100, pred=None, version='opt', thre=250):
    metrics = {}

    '''
    Threshold Independent
    '''
    grader = basic_metricor()
    # AUC_ROC, Precision, Recall, PointF1, PointF1PA, Rrecall, ExistenceReward, OverlapReward, Rprecision, RF, Precision_at_k = grader.metric_new(labels, score, pred, plot_ROC=False)
    AUC_ROC = grader.metric_ROC(labels, score)
    AUC_PR = grader.metric_PR(labels, score)

    # R_AUC_ROC, R_AUC_PR, _, _, _ = grader.RangeAUC(labels=labels, score=score, window=slidingWindow, plot_ROC=True)
    _, _, _, _, _, _,VUS_ROC, VUS_PR = generate_curve(labels.astype(int), score, slidingWindow, version, thre)


    '''
    Threshold Dependent
    if pred is None --> use the oracle threshold
    '''

    PointF1 = grader.metric_PointF1(labels, score, preds=pred)
    PointF1PA = grader.metric_PointF1PA(labels, score, preds=pred)
    EventF1PA = grader.metric_EventF1PA(labels, score, preds=pred)
    RF1 = grader.metric_RF1(labels, score, preds=pred)
    Affiliation_F = grader.metric_Affiliation(labels, score, preds=pred)

    metrics['AUC-PR'] = AUC_PR
    metrics['AUC-ROC'] = AUC_ROC
    metrics['VUS-PR'] = VUS_PR
    metrics['VUS-ROC'] = VUS_ROC

    metrics['Standard-F1'] = PointF1
    metrics['PA-F1'] = PointF1PA
    metrics['Event-based-F1'] = EventF1PA
    metrics['R-based-F1'] = RF1
    metrics['Affiliation-F'] = Affiliation_F
    return metrics


def get_metrics_and_best_threshold(score, labels, slidingWindow=100, version='opt', thre=250):
    metrics_dict = {}
    grader = basic_metricor()
    
    labels_int = labels.astype(int)

    # Threshold Independent
    AUC_ROC = grader.metric_ROC(labels_int, score)
    AUC_PR = grader.metric_PR(labels_int, score)
    # Note: Ensure generate_curve can handle labels_int if it expects a specific type
    _, _, _, _, _, _, VUS_ROC, VUS_PR = generate_curve(labels_int, score, slidingWindow, version, thre)

    # Get the best PointF1 and its corresponding threshold using the new method
    # This method determines the threshold based on maximizing PointF1 from scores
    PointF1, best_threshold = grader.metric_PointF1_with_threshold(labels_int, score, preds=None)

    # Generate predictions using this best_threshold
    if best_threshold is not None:
        # Ensure the comparison logic matches how the threshold was determined.
        # If metric_PointF1_with_threshold's optimal_threshold means "score > optimal_threshold", this is correct.
        # If it means "score >= optimal_threshold", this is also correct.
        # The typical sklearn precision_recall_curve thresholds are values such that scores strictly > threshold are positive.
        oracle_preds = (score > best_threshold).astype(int)
    else:
        # Fallback if threshold couldn't be determined
        # This case should be rare if metric_PointF1_with_threshold is robust
        # print(f"Warning: best_threshold was None for VUS calculation. Defaulting to all-zero predictions for dependent metrics.")
        oracle_preds = np.zeros_like(labels_int)


    # Calculate other threshold-dependent metrics using these oracle_preds
    PointF1PA = grader.metric_PointF1PA(labels_int, score, preds=oracle_preds)
    EventF1PA = grader.metric_EventF1PA(labels_int, score, preds=oracle_preds)
    RF1 = grader.metric_RF1(labels_int, score, preds=oracle_preds)
    Affiliation_F = grader.metric_Affiliation(labels_int, score, preds=oracle_preds)

    metrics_dict['AUC-PR'] = AUC_PR
    metrics_dict['AUC-ROC'] = AUC_ROC
    metrics_dict['VUS-PR'] = VUS_PR
    metrics_dict['VUS-ROC'] = VUS_ROC

    metrics_dict['Standard-F1'] = PointF1 # This is the max F1 obtained with best_threshold
    metrics_dict['PA-F1'] = PointF1PA
    metrics_dict['Event-based-F1'] = EventF1PA
    metrics_dict['R-based-F1'] = RF1
    metrics_dict['Affiliation-F'] = Affiliation_F
    
    return metrics_dict, best_threshold, oracle_preds


def get_metrics_pred(score, labels, pred, slidingWindow=100):
    metrics = {}

    grader = basic_metricor()

    PointF1 = grader.metric_PointF1(labels, score, preds=pred)
    PointF1PA = grader.metric_PointF1PA(labels, score, preds=pred)
    EventF1PA = grader.metric_EventF1PA(labels, score, preds=pred)
    RF1 = grader.metric_RF1(labels, score, preds=pred)
    Affiliation_F = grader.metric_Affiliation(labels, score, preds=pred)
    VUS_R, VUS_P, VUS_F = grader.metric_VUS_pred(labels, preds=pred, windowSize=slidingWindow)

    metrics['Standard-F1'] = PointF1
    metrics['PA-F1'] = PointF1PA
    metrics['Event-based-F1'] = EventF1PA
    metrics['R-based-F1'] = RF1
    metrics['Affiliation-F'] = Affiliation_F

    metrics['VUS-Recall'] = VUS_R
    metrics['VUS-Precision'] = VUS_P
    metrics['VUS-F'] = VUS_F

    return metrics
