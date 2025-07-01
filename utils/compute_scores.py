import json
import os
import time
import argparse
from collections import defaultdict
import numpy as np

EPSILON = 1e-8

class RunningAverage:
    def __init__(self):
        self.avg = 0
        self.count = 0

    def append(self, value):
        self.avg = (value + self.count * self.avg) / (self.count + 1)
        self.count += 1

    def get_value(self):
        return self.avg

class RunningAverageDict:
    """A dictionary of running averages."""

    def __init__(self):
        self._dict = dict(
            a1=RunningAverage(),
            a2=RunningAverage(),
            a3=RunningAverage(),
            abs_rel=RunningAverage(),
            rmse=RunningAverage(),
            log_10=RunningAverage(),
            rmse_log=RunningAverage(),
            silog=RunningAverage(),
            sq_rel=RunningAverage(),
        )

    def update(self, new_dict):
        if new_dict is None:
            return

        if self._dict is None:
            self._dict = dict()
            for key, value in new_dict.items():
                self._dict[key] = RunningAverage()

        for key, value in new_dict.items():
            self._dict[key].append(value)

    def get_value(self):
        if self._dict is None:
            return None
        return {key: value.get_value() for key, value in self._dict.items()}

def compute_errors(gt, pred):
    # Add EPSILON to denominators to avoid division by zero
    thresh = np.maximum((gt / (pred + EPSILON)), (pred / (gt + EPSILON)))
    a1 = (thresh < 1.10).mean()
    a2 = (thresh < 1.10**2).mean()
    a3 = (thresh < 1.10**3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / (gt + EPSILON))
    sq_rel = np.mean(((gt - pred) ** 2) / (gt + EPSILON))

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt + EPSILON) - np.log(pred + EPSILON)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred + EPSILON) - np.log(gt + EPSILON)
    # Handle potential negative values before sqrt
    silog_term = np.mean(err**2) - np.mean(err) ** 2
    silog = np.sqrt(np.maximum(silog_term, 0)) * 100

    log_10 = (np.abs(np.log10(gt + EPSILON) - np.log10(pred + EPSILON))).mean()
    return dict(
        a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_log=rmse_log, silog=silog, sq_rel=sq_rel
    )

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

def convert_number_word_to_digit(text):
    """Convert number words to digits."""
    if not isinstance(text, str):
        return str(text)
        
    number_words = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10'
    }
    text = text.lower().strip()
    for word, digit in number_words.items():
        text = text.replace(word, digit)
    return text

def extract_first_digit(text):
    """Extract the first digit from text."""
    import re
    # First convert number words to digits
    text = convert_number_word_to_digit(text)
    # Find first digit in the text
    match = re.search(r'\d+', text)
    if match:
        return match.group()
    return text

def normalize_qualitative_answer(answer, category):
    """Normalize qualitative answers based on category."""
    if not isinstance(answer, (str, int, float)):
        answer = str(answer)
    
    if category == 'left_right':
        # Handle common variations of left/right
        if not isinstance(answer, str):
            answer = str(answer)
        answer = answer.lower().strip()
        if answer in ['l', 'left', 'left side', 'left-hand', 'left hand']:
            return 'left'
        elif answer in ['r', 'right', 'right side', 'right-hand', 'right hand']:
            return 'right'
        return answer
    elif category == 'mcq':
        # Extract first digit after normalization
        return extract_first_digit(answer)
    return str(answer).strip()

def main():
    parser = argparse.ArgumentParser(description='Compute scores from normalized answers')
    parser.add_argument('--gt_path', help='Path to the ground truth normalized answers file')
    parser.add_argument('--pred_path', help='Path to the prediction normalized answers file')
    args = parser.parse_args()
    
    # Load data
    with open(args.gt_path) as f:
        gt_data = json.load(f)
    with open(args.pred_path) as f:
        pred_data = json.load(f)
    
    # Create sets of IDs for validation
    gt_ids = {item['id'] for item in gt_data}
    pred_ids = {item['id'] for item in pred_data}
    
    # Check for missing entries
    missing_in_pred = gt_ids - pred_ids
    missing_in_gt = pred_ids - gt_ids
    
    if missing_in_pred or missing_in_gt:
        print("\n===== WARNING: ID MISMATCH DETECTED =====")
        if missing_in_pred:
            print(f"\nMissing {len(missing_in_pred)} predictions in pred file:")
            for missing_id in sorted(missing_in_pred):
                print(f"  - {missing_id}")
        if missing_in_gt:
            print(f"\nMissing {len(missing_in_gt)} ground truth entries in gt file:")
            for missing_id in sorted(missing_in_gt):
                print(f"  - {missing_id}")
        print("\nProcessing will be skipped due to ID mismatch.")
        return
    
    # Create lookup dictionary for predictions
    pred_lookup = {item['id']: item['normalized_answer'] for item in pred_data}
    
    # Initialize result dictionaries
    qualitative_dict = defaultdict(list)
    quantitative_success_dict = defaultdict(list)
    quantitative_error_dict = defaultdict(list)
    final_metrics = defaultdict(RunningAverageDict)
    qualitative_id_dict = defaultdict(list)
    quantitative_id_dict = defaultdict(list)
    
    # Process each ground truth entry
    for gt_item in gt_data:
        question_id = gt_item['id']
        gt_answer = gt_item['normalized_answer']
        pred_answer = pred_lookup[question_id]
        
        category = gt_item['category']
        
        # Determine question type based on category
        if category in ['count', 'distance']:
            question_type = "quantitative"
        elif category in ['left_right', 'mcq']:
            question_type = "qualitative"
        
        if question_type == "quantitative":
            gt_value = float(gt_answer)
            try:
                pred_value = float(pred_answer)
            except ValueError:
                print(f"Invalid prediction value for question ID {question_id}: {pred_answer}")
                pred_value = 1e9
                
            # Compute success based on category
            if category == "count":
                gt_value = int(gt_value)
                pred_value = int(pred_value)
                success = gt_value == pred_value
                error_rate = abs(gt_value - pred_value) / max(1, gt_value)
            
            elif category == "distance":
                success = (pred_value <= (1.10 * gt_value)) and (
                    pred_value >= (0.90 * gt_value)
                )
                error_rate = (np.abs(pred_value - gt_value)) / (gt_value + EPSILON)
                if not success:
                    print(f"GT Value: {gt_value}, Pred Value: {pred_value}")
            
            # Store results
            quantitative_success_dict[category].append(int(success))
            quantitative_error_dict[category].append(error_rate)
            quantitative_id_dict[category].append(question_id)
            
            # Update detailed metrics
            final_metrics[category].update(compute_errors(np.array([gt_value])[None], np.array([pred_value])[None]))
                
                
        elif question_type == "qualitative":
            # Normalize answers based on category
            normalized_gt = normalize_qualitative_answer(gt_answer, category)
            normalized_pred = normalize_qualitative_answer(pred_answer, category)
            
            # For qualitative questions, we consider it correct if the normalized answers match
            success = int(normalized_gt == normalized_pred)
            qualitative_dict[category].append(success)
            qualitative_id_dict[category].append(question_id)
            
    # Calculate metrics and save results
    result_dict = {}
    
    # Calculate qualitative metrics
    total_qualitative = 0
    correct_qualitative = 0
    for qual_cat in qualitative_dict.keys():
        correct_qualitative += np.sum(qualitative_dict[qual_cat])
        total_qualitative += len(qualitative_dict[qual_cat])
        result_dict[f"Qual_{qual_cat}_acc"] = np.sum(qualitative_dict[qual_cat]) / (len(qualitative_dict[qual_cat])+EPSILON) * 100
    
    result_dict[f"Qual_overall_acc"] = correct_qualitative / (total_qualitative+EPSILON) * 100

    # Calculate quantitative metrics
    total_quantitative = 0
    correct_quantitative = 0
    accum_error = 0

    available_quant_cats = list(quantitative_success_dict.keys())
    
    for quant_cat in available_quant_cats:
        correct_quantitative += np.sum(quantitative_success_dict[quant_cat])
        result_dict[f"Quan_{quant_cat}_acc"] = (
            np.sum(quantitative_success_dict[quant_cat]) / (len(quantitative_success_dict[quant_cat])+EPSILON) * 100
        )
        total_quantitative += len(quantitative_success_dict[quant_cat])
        if quant_cat in quantitative_error_dict:
            accum_error += np.sum(quantitative_error_dict[quant_cat])
            result_dict[f"Quan_{quant_cat}_err"] = (
                np.sum(quantitative_error_dict[quant_cat]) / (len(quantitative_error_dict[quant_cat])+EPSILON) * 100
            )

    # Store detailed metrics in result dictionary
    for category in final_metrics:
        metrics_value = final_metrics[category].get_value()
        if metrics_value:
            formatted_metrics = {k: round(v, 3) for k, v in metrics_value.items()}
            result_dict[f"Quan_{category}_absrel"] = formatted_metrics.get("abs_rel", "N/A")
            result_dict[f"Quan_{category}_rmse"] = formatted_metrics.get("rmse", "N/A")
            result_dict[f"Quan_{category}_a1"] = formatted_metrics.get("a1", "N/A")

    # Print evaluation results
    print("\n===== EVALUATION RESULTS =====")
    
    # Print quantitative results
    print("\nQUANTITATIVE RESULTS:")
    for category in sorted(quantitative_success_dict.keys()):
        count = len(quantitative_success_dict[category])
        if count > 0:
            correct_count = sum(quantitative_success_dict[category])
            accuracy = (correct_count / count) * 100
            
            print(f"{category.capitalize()} ({count}): {correct_count}/{count} = {accuracy:.2f}%")
            
            if category in final_metrics and final_metrics[category].get_value():
                metrics_value = final_metrics[category].get_value()
                formatted_metrics = {k: round(v, 3) for k, v in metrics_value.items()}
                absrel = formatted_metrics.get("abs_rel", "N/A")
                print(f"  Abs Rel = {absrel}")
            
            if category in quantitative_error_dict and len(quantitative_error_dict[category]) > 0:
                err_rate = (
                    np.sum(quantitative_error_dict[category]) / len(quantitative_error_dict[category]) * 100
                )
                print(f"  Error Rate = {err_rate:.2f}%")
            
            result_dict[f"Quan_{category}_count"] = count
            result_dict[f"Quan_{category}_correct"] = correct_count
            result_dict[f"Quan_{category}_acc"] = accuracy

    # Print qualitative results
    print("\nQUALITATIVE RESULTS:")
    for category in sorted(qualitative_dict.keys()):
        count = len(qualitative_dict[category])
        if count > 0:
            correct_count = sum(qualitative_dict[category])
            accuracy = (correct_count / count) * 100
            
            print(f"{category} ({count}): {correct_count}/{count} = {accuracy:.2f}%")
            
            result_dict[f"Qual_{category}_count"] = count
            result_dict[f"Qual_{category}_correct"] = correct_count
            result_dict[f"Qual_{category}_acc"] = accuracy

    # Print overall summary
    print("\n===== OVERALL SUMMARY =====")
    
    # Define category weights
    category_weights = {
        'count': 0.25,
        'distance': 0.25,
        'left_right': 0.25,
        'mcq': 0.25
    }
    
    # Calculate weighted scores
    weighted_scores = {}
    total_weighted_score = 0.0
    total_weight = 0.0
    
    # Process quantitative categories
    for category in ['count', 'distance']:
        if category in quantitative_success_dict and len(quantitative_success_dict[category]) > 0:
            correct_count = sum(quantitative_success_dict[category])
            total_count = len(quantitative_success_dict[category])
            accuracy = (correct_count / total_count) * 100
            weighted_scores[category] = accuracy * category_weights[category]
            total_weighted_score += weighted_scores[category]
            total_weight += category_weights[category]
            print(f"{category.capitalize()} (weighted): {accuracy:.2f}% * {category_weights[category]:.2f} = {weighted_scores[category]:.2f}")
    
    # Process qualitative categories
    for category in ['left_right', 'mcq']:
        if category in qualitative_dict and len(qualitative_dict[category]) > 0:
            correct_count = sum(qualitative_dict[category])
            total_count = len(qualitative_dict[category])
            accuracy = (correct_count / total_count) * 100
            weighted_scores[category] = accuracy * category_weights[category]
            total_weighted_score += weighted_scores[category]
            total_weight += category_weights[category]
            print(f"{category} (weighted): {accuracy:.2f}% * {category_weights[category]:.2f} = {weighted_scores[category]:.2f}")
    
    # Calculate final weighted score
    if total_weight > 0:
        final_weighted_score = total_weighted_score / total_weight
        print(f"\nFinal Weighted Score: {final_weighted_score:.2f}%")
        result_dict["Final_weighted_score"] = final_weighted_score
    
    if total_quantitative > 0:
        quan_overall_acc = (correct_quantitative / total_quantitative) * 100
        print(f"Quantitative: {correct_quantitative}/{total_quantitative} = {quan_overall_acc:.2f}%")
        result_dict["Quan_total_count"] = total_quantitative
        result_dict["Quan_total_correct"] = correct_quantitative
        result_dict["Quan_overall_acc"] = quan_overall_acc
    else:
        print("Quantitative: No questions evaluated")

    if total_qualitative > 0:
        qual_overall_acc = (correct_qualitative / total_qualitative) * 100
        print(f"Qualitative: {correct_qualitative}/{total_qualitative} = {qual_overall_acc:.2f}%")
        result_dict["Qual_total_count"] = total_qualitative
        result_dict["Qual_total_correct"] = correct_qualitative
        result_dict["Qual_overall_acc"] = qual_overall_acc
    else:
        print("Qualitative: No questions evaluated")

    if (total_quantitative + total_qualitative) > 0:
        overall_acc = (correct_quantitative + correct_qualitative) / (total_quantitative + total_qualitative) * 100
        print(f"Overall: {correct_quantitative + correct_qualitative}/{total_quantitative + total_qualitative} = {overall_acc:.2f}%")
        result_dict["Total_count"] = total_quantitative + total_qualitative
        result_dict["Total_correct"] = correct_quantitative + correct_qualitative
        result_dict["Overall_acc"] = overall_acc

    ################# Only needed if you want to save results to file for debugging #################
    ################# by default we have GT and Pred contained in the same file ################
    try:
        # Use predictions folder instead of parent directory
        predictions_dir = os.path.join(os.path.dirname(args.gt_path), "predictions")
        os.makedirs(predictions_dir, exist_ok=True)
        
        # Create a timestamp for the output file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Convert numpy types to Python native types before saving
        result_dict_converted = convert_numpy_types(result_dict)
        
        # Create full results dictionary
        full_results = {
            "evaluation": result_dict_converted,
            "entries": []
        }
        
        # Create lookup dictionaries for success and error rates
        success_lookup = {}
        error_rate_lookup = {}
        
        # Populate lookup dictionaries
        for category in quantitative_success_dict:
            for i, success in enumerate(quantitative_success_dict[category]):
                question_id = quantitative_id_dict[category][i]
                success_lookup[question_id] = success
                error_rate_lookup[question_id] = quantitative_error_dict[category][i]
                
        for category in qualitative_dict:
            for i, success in enumerate(qualitative_dict[category]):
                question_id = qualitative_id_dict[category][i]
                success_lookup[question_id] = success
                error_rate_lookup[question_id] = 0.0 if success else 1.0
        
        # Add all entries with their evaluation results
        for gt_item in gt_data:
            question_id = gt_item['id']
            if question_id not in pred_lookup:
                continue
                
            entry = {
                "id": question_id,
                "image": gt_item.get("image", ""),
                "conversations": gt_item.get("conversations", []),
                "rle": gt_item.get("rle", []),
                "category": gt_item["category"],
                "ground_truth": {
                    "normalized_answer": gt_item["normalized_answer"],
                    "freeform_answer": gt_item.get("freeform_answer", "")
                },
                "prediction": {
                    "normalized_answer": pred_lookup[question_id]
                }
            }
            
            # Add evaluation result if available
            if question_id in success_lookup:
                entry["evaluation"] = {
                    "success": bool(success_lookup[question_id]),
                    "error_rate": float(error_rate_lookup[question_id])
                }
            
            full_results["entries"].append(entry)
        
        # Save the summary statistics
        result_dict_path = os.path.join(predictions_dir, f"score_{timestamp}.json")
        with open(result_dict_path, "w") as outfile:
            json.dump(result_dict_converted, outfile, indent=2)
        print(f"\nSaved summary results to: {result_dict_path}")
        
        # Save the full results
        full_results_path = os.path.join(predictions_dir, f"full_results_{timestamp}.json")
        with open(full_results_path, "w") as outfile:
            json.dump(full_results, outfile, indent=2)
        print(f"Saved full results to: {full_results_path}")
        
    except Exception as e:
        print(f"\nFailed to save results: {e}")

if __name__ == "__main__":
    main() 
