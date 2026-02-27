#!/usr/bin/env python3
"""
Master Script: Run Both Encoding Versions
Executes both label encoding and one-hot encoding analyses
"""

import subprocess
import os
import sys
from datetime import datetime

def run_analysis(version_name, script_path, output_dir):
    """Run an analysis version and capture results"""
    print("\n" + "="*80)
    print(f"Running {version_name} Analysis")
    print("="*80)
    print(f"Script: {script_path}")
    print(f"Output directory: {output_dir}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    # Change to the output directory
    original_dir = os.getcwd()
    os.chdir(output_dir)
    
    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print(f"\n✅ {version_name} completed successfully!")
        else:
            print(f"\n❌ {version_name} failed with return code {result.returncode}")
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"\n❌ Error running {version_name}: {e}")
        return False
    finally:
        # Return to original directory
        os.chdir(original_dir)

def main():
    """Run both versions of the analysis"""
    print("="*80)
    print("XGBoost + SHAP Analysis: Tanzania MHQ Study")
    print("Running Both Encoding Versions")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define versions
    versions = [
        {
            'name': 'Label Encoding',
            'script': os.path.join(base_dir, 'label_encoding_version', 'xgboost_shap_label_encoding.py'),
            'output': os.path.join(base_dir, 'label_encoding_version')
        },
        {
            'name': 'One-Hot Encoding',
            'script': os.path.join(base_dir, 'one_hot_encoding_version', 'xgboost_shap_one_hot_encoding.py'),
            'output': os.path.join(base_dir, 'one_hot_encoding_version')
        }
    ]
    
    # Track results
    results = {}
    
    # Run each version
    for version in versions:
        success = run_analysis(
            version['name'],
            version['script'],
            version['output']
        )
        results[version['name']] = success
    
    # Summary
    print("\n" + "="*80)
    print("EXECUTION SUMMARY")
    print("="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nResults by version:")
    for name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {name}: {status}")
    
    print("\n📁 Output Locations:")
    print(f"  • Label Encoding: {os.path.join(base_dir, 'label_encoding_version')}")
    print(f"  • One-Hot Encoding: {os.path.join(base_dir, 'one_hot_encoding_version')}")
    
    all_success = all(results.values())
    if all_success:
        print("\n🎉 All analyses completed successfully!")
        print("\n📊 Next steps:")
        print("  1. Review visualizations in each version's folder")
        print("  2. Compare results between versions")
        print("  3. See COMPARISON_LABEL_VS_ONEHOT.md for detailed comparison")
    else:
        print("\n⚠️  Some analyses failed. Check error messages above.")
    
    print("="*80 + "\n")
    
    return 0 if all_success else 1

if __name__ == "__main__":
    exit(main())
