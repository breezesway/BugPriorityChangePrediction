#!/usr/bin/env python3
"""
LLM-based Bug Priority Change Prediction - Phase 2

This script uses large language models (ChatGPT-5 and Gemini 2.5 Pro) to predict
the specific priority level that bugs will change to in their lifecycle based on 
initial bug reports and development process information.
"""

import json
import pandas as pd
import openai
import google.generativeai as genai
from sklearn.metrics import f1_score
import os
from typing import List, Dict, Tuple
import time

class LLMPriorityLevelPredictor:
    def __init__(self):
        """Initialize the LLM predictor with API configurations."""
        # OpenAI API configuration
        self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Google Gemini API configuration
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        self.gemini_model = genai.GenerativeModel('gemini-2.5-pro')
        
        # Priority mapping
        self.priority_mapping = {
            'Blocker': 0,
            'Critical': 1, 
            'Major': 2,
            'Minor': 3,
            'Trivial': 4
        }
        
        # Reverse mapping for output
        self.reverse_priority_mapping = {v: k for k, v in self.priority_mapping.items()}
    
    def load_issues(self, json_file_path: str) -> List[Dict]:
        """Load issues from JSON file."""
        with open(json_file_path, 'r', encoding='utf-8') as f:
            issues = json.load(f)
        print(f"Loaded {len(issues)} issues from {json_file_path}")
        return issues
    
    def load_ground_truth(self, csv_file_path: str) -> Dict[str, str]:
        """Load ground truth labels from CSV/Excel file."""

        df = pd.read_csv(csv_file_path)
        print(f"Loaded ground truth from CSV file: {csv_file_path}")
        
        # Create mapping from issue key to final priority
        ground_truth = {}
        for _, row in df.iterrows():
            if 'FinalPriority' in df.columns:
                ground_truth[row['key']] = row['TargetPriority']
        
        print(f"Loaded ground truth for {len(ground_truth)} issues")
        return ground_truth
    
    def create_prompt(self, issues: List[Dict]) -> str:
        """Create optimized prompt for LLM prediction."""
        prompt = """In the issue tracking system JIRA, each bug has a priority to indicate the urgency of the bug. There are 5 priority categories, from high to low: Blocker, Critical, Major, Minor, Trivial.

When a bug is reported, developers assign it a priority. However, the priority may change during the bug's lifecycle. Such changes may be influenced by various factors, including:

**Key Factors Affecting Priority Changes:**
1. **Bug severity and impact scope**: Number of affected users, system stability, data security, etc.
2. **Detail and clarity of bug description**: More detailed descriptions lead to more accurate priority assessment
3. **Bug complexity and fix difficulty**: Technical complexity, dependencies, testing difficulty, etc.
4. **Project time pressure and resource allocation**: Release cycles, team resources, business priorities
5. **Experience and authority of bug reporter**: Reports from senior developers are usually more accurate
6. **Feedback during development process**: Comments, historical changes, related links provide additional information
7. **Business impact and user feedback**: Bugs that directly affect user experience or business processes
8. **Technical debt and architectural impact**: Impact on system architecture, performance, maintainability

**General Patterns of Priority Changes:**
- **Upgrade**: Found to be more severe, affecting more users, requiring more urgent fixes
- **Downgrade**: Found to have smaller impact, alternative solutions available, can be deferred
- **No change**: Initial assessment was accurate, no adjustment needed

Now I provide you with a JSON file containing detailed information about 100 bugs, including:
- key: Unique identifier of the bug
- summary: Bug summary
- description: Detailed bug description
- FromPriority: Initially assigned priority
- project: Project it belongs to
- issuetype: Bug type
- labels: Label information
- comments: Comment information during development process
- histories: Historical change records
- Other related metadata

Please analyze the characteristics of each bug and the information from the development process based on the information I provide, and predict which priority level these 100 bugs will change to.

**Output Requirements:**
Please strictly follow the following format for output, one line per bug:
Key: [bug key]
Prediction: [Blocker/Critical/Major/Minor/Trivial]

**Notes:**
- Only output one of the 5 standard priority levels
- If the bug priority will not change, please output the same level as FromPriority
- Make judgments based on factors such as bug severity, impact scope, technical complexity
- Consider comment and historical change information from the development process

---

Bug data:
"""
        
        # Add issues data to prompt
        prompt += json.dumps(issues, ensure_ascii=False, indent=2)
        
        return prompt
    
    def parse_llm_response(self, response: str) -> Dict[str, str]:
        """Parse LLM response to extract predictions."""
        predictions = {}
        lines = response.split('\n')
        
        current_key = None
        for line in lines:
            line = line.strip()
            if line.startswith('Key:'):
                current_key = line.replace('Key:', '').strip()
            elif line.startswith('Prediction:') and current_key:
                prediction_str = line.replace('Prediction:', '').strip()
                # Validate that it's one of the expected priorities
                if prediction_str in self.priority_mapping:
                    predictions[current_key] = prediction_str
                current_key = None
        
        return predictions
    
    def predict_with_chatgpt5(self, prompt: str) -> Dict[str, str]:
        """Make prediction using ChatGPT-5."""
        try:
            print("Making prediction with ChatGPT-5...")
            response = self.openai_client.chat.completions.create(
                model="gpt-5",  # Note: This might need to be adjusted based on actual model name
                messages=[
                    {"role": "system", "content": "You are an expert in software engineering and bug tracking systems. Analyze the provided bug reports and predict the final priority level they will have."},
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
    
    def predict_with_gemini(self, prompt: str) -> Dict[str, str]:
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
    
    def calculate_metrics(self, predictions: Dict[str, str], ground_truth: Dict[str, str]) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        # Find common keys
        common_keys = set(predictions.keys()) & set(ground_truth.keys())
        
        if not common_keys:
            print("No common keys found between predictions and ground truth")
            return {}
        
        # Extract predictions and ground truth for common keys
        y_pred = [self.priority_mapping[predictions[key]] for key in common_keys]
        y_true = [self.priority_mapping[ground_truth[key]] for key in common_keys]
        
        # Calculate F1-weighted and F1-macro
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        metrics = {
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro
        }
        
        print(f"Evaluated on {len(common_keys)} common issues")
        return metrics
    
    def run_prediction(self, json_file_path: str, csv_file_path: str):
        """Run the complete prediction pipeline."""
        print("Starting LLM-based Bug Priority Level Prediction - Phase 2")
        print("=" * 70)
        
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
            print(f"\n{'='*25} {model_name} {'='*25}")
            
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
                    print(f"F1-weighted: {metrics['f1_weighted']:.4f}")
                    print(f"F1-macro:    {metrics['f1_macro']:.4f}")
                else:
                    print(f"No metrics calculated for {model_name}")
            else:
                print(f"No predictions obtained from {model_name}")
            
            # Add delay between API calls to avoid rate limiting
            time.sleep(2)
        
        # Summary
        print(f"\n{'='*70}")
        print("FINAL RESULTS SUMMARY - PHASE 2")
        print(f"{'='*70}")
        
        if results:
            for model_name, result in results.items():
                if 'metrics' in result:
                    metrics = result['metrics']
                    print(f"\n{model_name} Performance:")
                    print(f"  F1-weighted: {metrics['f1_weighted']:.4f}")
                    print(f"  F1-macro:    {metrics['f1_macro']:.4f}")
                    print(f"  {'-'*50}")
        else:
            print("No results obtained from any model.")
        
        return results

def main():
    """Main function to run the prediction."""
    # File paths
    json_file = "./code/analysis_in_discussion/llms_explore/rawIssue2.json"
    csv_file = "./data/phase_2.csv"
    
    # Check if API keys are set
    if not os.getenv('OPENAI_API_KEY'):
        print("Warning: OPENAI_API_KEY environment variable not set")
    if not os.getenv('GEMINI_API_KEY'):
        print("Warning: GEMINI_API_KEY environment variable not set")
    
    # Initialize predictor and run
    predictor = LLMPriorityLevelPredictor()
    results = predictor.run_prediction(json_file, csv_file)

if __name__ == "__main__":
    main()
