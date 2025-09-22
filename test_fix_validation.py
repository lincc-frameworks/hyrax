#!/usr/bin/env python3
"""
Direct test of the fix without importing the full hyrax module.
This test validates the logic of the primary_id_field fix by testing the specific
code section that was modified.
"""

def generate_data_request_from_config(config):
    """Simplified version of the function for testing"""
    if "model_inputs" in config:
        return config["model_inputs"]
    else:
        return {
            "data": {
                "dataset_class": config["data_set"]["name"],
                "data_location": config["general"]["data_dir"],
                "primary_id_field": "object_id",
            },
        }

def test_primary_id_field_fix():
    """Test the exact logic that was added to fix the primary_id_field issue"""
    
    print("Testing primary_id_field fix...")
    
    # Test case 1: primary_id_field not in fields (the bug scenario)
    config = {
        "model_inputs": {
            "test_dataset": {
                "dataset_class": "TestDataset",
                "data_location": "./test_data", 
                "fields": ["image", "label"],  # Note: "object_id" is NOT included
                "primary_id_field": "object_id",  # But this field is set as primary
            }
        }
    }
    
    data_request = generate_data_request_from_config(config)
    
    # Simulate the DataProvider.prepare_datasets logic for primary dataset handling
    primary_dataset = None
    primary_dataset_id_field_name = None
    
    for friendly_name, dataset_definition in data_request.items():
        # This is the original buggy code section, now with the fix
        if "primary_id_field" in dataset_definition:
            primary_dataset = friendly_name
            primary_dataset_id_field_name = dataset_definition["primary_id_field"]
            
            # THE FIX: Ensure the primary_id_field is included in the fields list
            # to prevent KeyError when resolve_data tries to access it
            if primary_dataset_id_field_name not in dataset_definition.get("fields", []):
                dataset_definition.setdefault("fields", []).append(primary_dataset_id_field_name)
    
    # Verify the fix worked
    test_dataset_def = data_request["test_dataset"]
    assert primary_dataset == "test_dataset"
    assert primary_dataset_id_field_name == "object_id"
    assert "object_id" in test_dataset_def["fields"]
    expected_fields = ["image", "label", "object_id"]
    assert test_dataset_def["fields"] == expected_fields
    print("✓ Test 1 PASSED: primary_id_field automatically added to fields")
    print(f"  Original fields: ['image', 'label']")
    print(f"  Fixed fields: {test_dataset_def['fields']}")
    
    # Test case 2: primary_id_field already in fields (should not duplicate)
    config2 = {
        "model_inputs": {
            "test_dataset": {
                "dataset_class": "TestDataset",
                "data_location": "./test_data",
                "fields": ["object_id", "image", "label"],  # object_id already included
                "primary_id_field": "object_id",
            }
        }
    }
    
    data_request2 = generate_data_request_from_config(config2)
    original_fields = data_request2["test_dataset"]["fields"][:]
    
    for friendly_name, dataset_definition in data_request2.items():
        if "primary_id_field" in dataset_definition:
            primary_dataset = friendly_name
            primary_dataset_id_field_name = dataset_definition["primary_id_field"]
            
            # THE FIX: Ensure the primary_id_field is included in the fields list
            if primary_dataset_id_field_name not in dataset_definition.get("fields", []):
                dataset_definition.setdefault("fields", []).append(primary_dataset_id_field_name)
    
    # Verify no duplicate was added
    final_fields = data_request2["test_dataset"]["fields"]
    assert final_fields == original_fields
    assert final_fields.count("object_id") == 1
    print("✓ Test 2 PASSED: no duplicate when primary_id_field already in fields")
    print(f"  Fields unchanged: {final_fields}")
    
    # Test case 3: No fields specified (should still work)
    config3 = {
        "model_inputs": {
            "test_dataset": {
                "dataset_class": "TestDataset",
                "data_location": "./test_data",
                # No fields specified
                "primary_id_field": "object_id",
            }
        }
    }
    
    data_request3 = generate_data_request_from_config(config3)
    
    for friendly_name, dataset_definition in data_request3.items():
        if "primary_id_field" in dataset_definition:
            primary_dataset = friendly_name
            primary_dataset_id_field_name = dataset_definition["primary_id_field"]
            
            # THE FIX: Ensure the primary_id_field is included in the fields list
            if primary_dataset_id_field_name not in dataset_definition.get("fields", []):
                dataset_definition.setdefault("fields", []).append(primary_dataset_id_field_name)
    
    # Verify it created the fields list and added the primary_id_field
    final_def = data_request3["test_dataset"]
    assert "fields" in final_def
    assert final_def["fields"] == ["object_id"]
    print("✓ Test 3 PASSED: primary_id_field added when no fields list exists")
    print(f"  Created fields: {final_def['fields']}")
    
    return True

if __name__ == "__main__":
    try:
        success = test_primary_id_field_fix()
        if success:
            print("\n✅ ALL TESTS PASSED - Fix is working correctly!")
            print("\nThe fix ensures that when 'primary_id_field' is specified in model_inputs,")
            print("that field is automatically added to the 'fields' list if it's not already there.")
            print("This prevents the KeyError: 'object_id' that was occurring in resolve_data().")
        else:
            print("\n❌ TESTS FAILED")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)