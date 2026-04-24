# Outdated Issues Review Report
**lincc-frameworks/hyrax** | Generated: 2026-04-12

---

## Executive Summary

Analysis of 92 open issues identified **11 stale issues** (4+ months old with no activity). Of these:
- **5 issues are likely outdated** and can be closed
- **3 investigations have been completed** and can be closed/documented
- **2 issues reflect valid design decisions** that may need clarification
- **2 issues remain relevant** but are lower priority

---

## Issues Recommended for Closure

### ✅ #116: "Consider allowing user to specify RNG seed in config" (Nov 12, 2024)
**Status:** ALREADY IMPLEMENTED  
**Finding:** The feature is already fully implemented and functional:
- Seed is configurable via `config["data_set"]["seed"]`
- Used for torch.manual_seed(), np.random.seed(), and dataset shuffling
- Random datasets also support seed configuration

**Action:** Close with note that feature is already available

---

### ✅ #133: "Default implementation of train_step for models" (Dec 7, 2024)
**Status:** ALREADY IMPLEMENTED  
**Finding:** The framework already provides this through the `train_batch()` interface:
- All registered models implement `train_batch()` method
- pytorch_ignite.py uses `create_engine("train_batch", ...)` as standard pattern
- Models also implement `validate_batch()` method

**Action:** Close with documentation pointer to existing pattern

---

### ✅ #124: "Ctrl-C while downloading HSC data produces stack trace" (Dec 2, 2024)
**Status:** LIKELY FIXED  
**Finding:** Modern exception handling is now in place:
- download.py line 178: `finally:` block ensures manifest is written even on KeyboardInterrupt
- downloadCutout.py has explicit KeyboardInterrupt handling with logging
- Graceful cleanup implemented in download thread management

**Action:** Close with confirmation that graceful interrupt handling is now in place

---

### ✅ #20: "Investigate ONNX as a tool for working with multiple ML frameworks" (Aug 14, 2024)
**Status:** INVESTIGATION COMPLETE  
**Finding:** ONNX support is fully implemented:
- `/src/hyrax/model_exporters.py` contains ONNX export implementation
- Supports PyTorch to ONNX conversion with onnxruntime validation
- Framework extensibility designed for other frameworks

**Action:** Close as investigation complete with documentation of ONNX capabilities

---

### ✅ #72: "Investigate using HF Datasets class" (Sep 24, 2024)
**Status:** INVESTIGATION COMPLETE / IMPLEMENTED  
**Finding:** Feature is fully implemented and merged:
- PR #846 merged MultimodalUniverseDataset with HuggingFace datasets support
- `/src/hyrax/datasets/mmu_dataset.py` fully functional
- Supports `hf://` URI scheme with max_samples and split configuration

**Action:** Close as investigation complete with reference to MMU dataset implementation

---

### ✅ #96: "Investigate interactive 3d plotting tools" (Oct 18, 2024)
**Status:** INVESTIGATION COMPLETE / IMPLEMENTED  
**Finding:** Interactive 3D visualization is fully implemented:
- `/src/hyrax/3d_viz/` directory contains complete implementation
- Uses Plotly for interactive 3D scatter plots
- `plot_umap_3d_interactive()` provides interactive 3D UMAP visualization
- Custom visualization server available for real-time interaction

**Action:** Close as investigation complete with reference to 3D viz module

---

### ✅ #11: "Reconsider the location of fibad_cli" (Aug 12, 2024)
**Status:** LIKELY OUTDATED (Architecture Changed)  
**Finding:** Project has evolved away from FIBAD integration:
- FIBAD only mentioned in HSCDataset documentation
- Main CLI is now `hyrax_cli` in `/src/hyrax_cli/main.py`
- HSC downloads handled through Hyrax's own downloader infrastructure
- Recent commits show focus on Hyrax as independent framework

**Action:** Close as architecture has implicitly resolved this

---

## Issues Requiring Design Review

### ⚠️ #58: "Downloader support for variable filter, sh, sw, rerun, type" (Sep 6, 2024)
**Status:** PARTIALLY ADDRESSED / DESIGN DECISION  
**Finding:** 
- Current implementation: `VARIABLE_FIELDS = ["tract", "ra", "dec"]` (catalog-driven)
- Filter, sh, sw, rerun, type are available as fixed config parameters
- Design distinguishes between per-row variable fields vs. fixed config parameters
- These parameters ARE available in downloadCutout.py but as config defaults, not per-row variables

**Action:** Clarify design intent with team
- Is the current partial implementation sufficient?
- Should these become per-row variable fields like tract/ra/dec?
- Or is the config-based approach the intended design?

**Recommendation:** Schedule brief design discussion to confirm intent

---

### ⚠️ #63: "Support downloads where tract, patch and object id are not present" (Sep 13, 2024)
**Status:** LIKELY OUTDATED (Design Decision)  
**Finding:**
- System is designed with required fields: `object_id` and `tract` in manifest
- These are extracted from catalog FITS file (mandatory)
- Supporting optional identifiers would require significant refactoring of manifest system
- Current design ensures data traceability

**Action:** Clarify requirements
- Is there a use case for optional identifiers?
- Or is the current requirement appropriate for data integrity?

**Recommendation:** Close unless there's a strong production use case for optional identifiers

---

## Issues Remaining Relevant

### 📋 #64: "Write a perf test for training" (Sep 13, 2024)
**Status:** STILL RELEVANT (Not Addressed)  
**Finding:**
- No dedicated performance tests in test suite
- 36 existing test files are functional/unit focused
- Recent development of training infrastructure (pytorch-ignite, model_exporters) suggests value
- Active evolution of training capabilities shown in recent commits

**Action:** Keep for future sprint
- Low priority but would be valuable as performance becomes critical
- Schedule when benchmarking requirements emerge

---

### 📋 #142: "Need a simple data_set validator script" (Dec 16, 2024)
**Status:** STILL RELEVANT (Not Addressed)  
**Finding:**
- No dedicated validator script currently exists
- Framework has strong built-in config validation (pydantic models)
- PR #853 suggests ongoing documentation/validation improvements
- Would be UX improvement but not blocking functionality

**Action:** Keep as low-priority enhancement
- Consider for next quality-of-life improvement cycle
- Could provide external validation utility for data_set configs

---

## Summary Statistics

| Category | Count | Details |
|----------|-------|---------|
| Recommended for Closure | 7 | #11, #20, #72, #96, #116, #124, #133 |
| Needs Design Review | 2 | #58, #63 |
| Still Relevant | 2 | #64, #142 |
| **Total Stale Issues Analyzed** | **11** | 4+ months old, no activity |

---

## Overall Repository Health

**Positive Signals:**
- Recent activity shows strong development momentum (last PR from Apr 11, 2026)
- Investigation issues have been properly completed and implemented
- Architecture is mature with clear patterns and interfaces
- Configuration system is robust (Pydantic validation)

**Areas for Improvement:**
- 39 issues (42%) are unlabeled - adding labels would improve issue management
- No milestones assigned to any issues - consider adding for release planning
- All 92 issues are unassigned - consider assigning owners to high-priority items

---

## Recommended Next Steps

1. **Immediate (closure candidates):** Review and close #11, #20, #72, #96, #116, #124, #133
2. **Design review:** Schedule brief meeting to clarify #58 and #63
3. **Long-term:** Keep #64 and #142 for future work as time permits
4. **Housekeeping:** Consider adding labels to the 39 unlabeled issues for better organization

