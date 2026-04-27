# Test Checklist

Cover the smallest representative sample:

- Constructor loads or connects using temporary sample data.
- `len(dataset)` matches the number of objects.
- `get_<primary_id_field>(idx)` returns stable unique IDs.
- Each requested getter returns the expected value, shape, and type.
- Dataset-specific config defaults are read with bracket access.
- Pass-through kwargs reach the underlying library when supported.
- Missing `data_location` raises only when the dataset cannot sensibly operate without it.

Prefer tests that create realistic tiny data in `tmp_path` over tests that depend on repository data fixtures. Keep the assertions tied to the public Hyrax behavior: constructor, `__len__`, and `get_<field>` methods.

Run the targeted dataset tests first. Run broader tests only when the change touches shared dataset loading, config parsing, data request handling, or collation behavior.
