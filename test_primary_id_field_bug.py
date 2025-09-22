#!/usr/bin/env python3
"""
Test script to reproduce the KeyError when primary_id_field is not in fields list.
This is a standalone test to verify the bug exists and will be fixed.
"""
import sys
import os

# Add the src directory to the path so we can import hyrax
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_primary_id_field_not_in_fields():
    """
    Test to reproduce the bug where primary_id_field not in fields causes KeyError.
    """
    try:
        # Import within the function to handle potential import issues
        import hyrax
        from hyrax.data_sets.data_provider import DataProvider
        
        h = hyrax.Hyrax()
        
        # Add the required HyraxRandomDataset config
        h.config["data_set"]["HyraxRandomDataset"] = {
            "size": 10,
            "shape": [2, 3, 3],
            "seed": 42,
            "provided_labels": ["cat", "dog"],
            "number_invalid_values": 0,
            "invalid_value_type": "nan",
        }
        
        # Configure a dataset where primary_id_field is NOT in the fields list
        # This should cause the KeyError described in the issue
        model_inputs = {
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": "./test_data",
                "fields": ["image", "label"],  # Note: "object_id" is NOT included
                "primary_id_field": "object_id",  # But this field is set as primary
            }
        }
        
        h.config["model_inputs"] = model_inputs
        
        # This should work fine - no error during preparation
        dp = DataProvider(h.config)
        
        # This should trigger the KeyError: 'object_id'
        # because object_id is not in the fields list but is expected in resolve_data
        try:
            data = dp.resolve_data(0)
            print("ERROR: Expected KeyError but got data:", data)
            return False
        except KeyError as e:
            print(f"SUCCESS: Reproduced the bug - KeyError: {e}")
            return True
            
    except Exception as e:
        print(f"ERROR: Unexpected exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_primary_id_field_not_in_fields()
    if success:
        print("Bug reproduction test PASSED - KeyError was reproduced as expected")
    else:
        print("Bug reproduction test FAILED - KeyError was not reproduced")
    sys.exit(0 if success else 1)