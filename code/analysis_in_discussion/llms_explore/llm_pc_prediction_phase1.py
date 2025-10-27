#!/usr/bin/env python3
"""
LLM-based Bug Priority Change Prediction

This script uses large language models (ChatGPT-5 and Gemini 2.5 Pro) to predict
whether bug priorities will change in their lifecycle based on initial bug reports.
"""

import json
import pandas as pd
import openai
import google.generativeai as genai
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
from typing import List, Dict, Tuple
import time

class LLMPriorityPredictor:
    def __init__(self):
        """Initialize the LLM predictor with API configurations."""
        # OpenAI API configuration
        self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Google Gemini API configuration
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        self.gemini_model = genai.GenerativeModel('gemini-2.5-pro')
        
        # Priority mapping
        self.priority_mapping = {
            'Blocker': 1,
            'Critical': 2, 
            'Major': 3,
            'Minor': 4,
            'Trivial': 5
        }
    
    def load_issues(self, json_file_path: str) -> List[Dict]:
        """Load issues from JSON file."""
        with open(json_file_path, 'r', encoding='utf-8') as f:
            issues = json.load(f)
        print(f"Loaded {len(issues)} issues from {json_file_path}")
        return issues
    
    def load_ground_truth(self, csv_file_path: str) -> Dict[str, bool]:
        """Load ground truth labels from CSV file."""
        df = pd.read_csv(csv_file_path)
        # Create mapping from issue key to change status
        ground_truth = {}
        for _, row in df.iterrows():
            ground_truth[row['key']] = bool(row['Changed'])
        print(f"Loaded ground truth for {len(ground_truth)} issues")
        return ground_truth
    
    def create_prompt(self, issues: List[Dict]) -> str:
        """Create optimized prompt for LLM prediction."""
        prompt = """In the issue tracking system JIRA, each bug has a priority to indicate the urgency of the bug. There are 5 priority categories, from high to low: Blocker, Critical, Major, Minor, Trivial.

When a bug is reported, developers assign it a priority. However, the priority may change during the bug's lifecycle. Such changes may be influenced by various factors, including:
- The severity and impact scope of the bug
- The detail and clarity of the bug description
- The complexity and difficulty of fixing the bug
- Project time pressure and resource allocation
- The experience and authority of the bug reporter

Now I provide you with a JSON file containing detailed information about 200 bugs when they were reported, including:
- key: Unique identifier of the bug
- summary: Bug summary
- description: Detailed bug description
- Priority: Initially assigned priority
- project: Project it belongs to
- issuetype: Bug type
- labels: Label information
- Other related metadata

Please analyze the characteristics of each bug based on the information I provide and predict whether the priority of these 200 bugs will change in their subsequent lifecycle.

Please output the results in the following format:
Key: [bug key]
Prediction: [True/False]
Reason: [Brief explanation of prediction reason]

---

Bug data:
"""
        
        # Add issues data to prompt
        prompt += json.dumps(issues, ensure_ascii=False, indent=2)
        
        return prompt
    
    def parse_llm_response(self, response: str) -> Dict[str, bool]:
        """Parse LLM response to extract predictions."""
        predictions = {}
        lines = response.split('\n')
        
        current_key = None
        for line in lines:
            line = line.strip()
            if line.startswith('Key:'):
                current_key = line.replace('Key:', '').strip()
            elif line.startswith('Prediction:') and current_key:
                prediction_str = line.replace('Prediction:', '').strip().lower()
                if prediction_str == 'true':
                    predictions[current_key] = True
                elif prediction_str == 'false':
                    predictions[current_key] = False
                current_key = None
        
        return predictions
    
    def predict_with_chatgpt5(self, prompt: str) -> Dict[str, bool]:
        """Make prediction using ChatGPT-5."""
        try:
            print("Making prediction with ChatGPT-5...")
            response = self.openai_client.chat.completions.create(
                model="gpt-5",  # Note: This might need to be adjusted based on actual model name
                messages=[
                    {"role": "system", "content": "You are an expert in software engineering and bug tracking systems. Analyze the provided bug reports and predict whether their priorities will change."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=4000
            )
            
            response_text = response.choices[0].message.content
            predictions = self.parse_llm_response(response_text)
            print(f"ChatGPT-5 predictions: {len(predictions)} issues processed")
            return predictions
            
        except Exception as e:
            print(f"Error with ChatGPT-5: {e}")
            return {}
    
    def predict_with_gemini(self, prompt: str) -> Dict[str, bool]:
        """Make prediction using Gemini 2.5 Pro."""
        try:
            print("Making prediction with Gemini 2.5 Pro...")
            response = self.gemini_model.generate_content(prompt)
            response_text = response.text
            predictions = self.parse_llm_response(response_text)
            print(f"Gemini predictions: {len(predictions)} issues processed")
            return predictions
            
        except Exception as e:
            print(f"Error with Gemini: {e}")
            return {}
    
    def calculate_metrics(self, predictions: Dict[str, bool], ground_truth: Dict[str, bool]) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        # Find common keys
        common_keys = set(predictions.keys()) & set(ground_truth.keys())
        
        if not common_keys:
            print("No common keys found between predictions and ground truth")
            return {}
        
        # Extract predictions and ground truth for common keys
        y_pred = [predictions[key] for key in common_keys]
        y_true = [ground_truth[key] for key in common_keys]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        }
        
        print(f"Evaluated on {len(common_keys)} common issues")
        return metrics
    
    def run_prediction(self, json_file_path: str, csv_file_path: str):
        """Run the complete prediction pipeline."""
        print("Starting LLM-based Bug Priority Change Prediction")
        print("=" * 60)
        
        # Load data
        issues = self.load_issues(json_file_path)
        ground_truth = self.load_ground_truth(csv_file_path)
        
        # Create prompt
        prompt = self.create_prompt(issues)
        print(f"Created prompt with {len(prompt)} characters")
        
        # Models to test
        models = [
            ("ChatGPT-5", self.predict_with_chatgpt5),
            ("Gemini 2.5 Pro", self.predict_with_gemini)
        ]
        
        results = {}
        
        for model_name, predict_func in models:
            print(f"\n{'='*20} {model_name} {'='*20}")
            
            # Make predictions
            predictions = predict_func(prompt)
            
            if predictions:
                # Calculate metrics
                metrics = self.calculate_metrics(predictions, ground_truth)
                
                if metrics:
                    results[model_name] = {
                        'predictions': predictions,
                        'metrics': metrics
                    }
                    
                    print(f"\n{model_name} Results:")
                    print(f"Accuracy:  {metrics['accuracy']:.4f}")
                    print(f"Precision: {metrics['precision']:.4f}")
                    print(f"Recall:    {metrics['recall']:.4f}")
                    print(f"F1 Score:  {metrics['f1']:.4f}")
                else:
                    print(f"No metrics calculated for {model_name}")
            else:
                print(f"No predictions obtained from {model_name}")
            
            # Add delay between API calls to avoid rate limiting
            time.sleep(2)
        
        # Summary
        print(f"\n{'='*60}")
        print("FINAL RESULTS SUMMARY")
        print(f"{'='*60}")
        
        if results:
            for model_name, result in results.items():
                if 'metrics' in result:
                    metrics = result['metrics']
                    print(f"\n{model_name} Performance:")
                    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
                    print(f"  Precision: {metrics['precision']:.4f}")
                    print(f"  Recall:    {metrics['recall']:.4f}")
                    print(f"  F1 Score:  {metrics['f1']:.4f}")
                    print(f"  {'-'*40}")
        else:
            print("No results obtained from any model.")
        
        return results

def main():
    """Main function to run the prediction."""
    # File paths
    json_file = "./code/analysis_in_discussion/llms_explore/rawIssue1.json"
    csv_file = "./data/phase_1.csv"
    
    # Check if API keys are set
    if not os.getenv('OPENAI_API_KEY'):
        print("Warning: OPENAI_API_KEY environment variable not set")
    if not os.getenv('GEMINI_API_KEY'):
        print("Warning: GEMINI_API_KEY environment variable not set")
    
    # Initialize predictor and run
    predictor = LLMPriorityPredictor()
    results = predictor.run_prediction(json_file, csv_file)

if __name__ == "__main__":
    main()
