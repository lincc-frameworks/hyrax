
# Feature Specification: Flexible Dataset Splitting for Model Training

**Feature Branch**: `001-build-a-flexible`  
**Created**: 2025-10-13  
**Status**: Draft  
**Input**: Build a flexible system that allows a user to declare the method of splitting a data set for training a model into training and validation sets. Two modes should be supported to start with. The first should be "all data in one location", meaning that the user would need to declare percentages or absolute numbers to define the splits (this is already implemented). The second should be "data in separate directories", meaning that the user has already places files for training and validation sets in distinct directories. The user should still be able to request data from multiple datasets using the "model_inputs" configuration table. Ideally they would be able to define their style of split in a ergonomic way that doesn't require lots of duplicated configuration settings, and is also very intuitive.

## User Scenarios & Testing *(mandatory)*

<!--
  IMPORTANT: User stories should be PRIORITIZED as user journeys ordered by importance.
  Each user story/journey must be INDEPENDENTLY TESTABLE - meaning if you implement just ONE of them,
  you should still have a viable MVP (Minimum Viable Product) that delivers value.
  
  Assign priorities (P1, P2, P3, etc.) to each story, where P1 is the most critical.
  Think of each story as a standalone slice of functionality that can be:
  - Developed independently
  - Tested independently
  - Deployed independently
  - Demonstrated to users independently
-->

### User Story 1 - Select Split Method Ergonomically (Priority: P1)
A user configures a model training run and selects the method for splitting their dataset (either "all data in one location" or "data in separate directories") using an intuitive configuration option.

**Why this priority**: This is the core user flow; it enables flexible, user-driven dataset splitting for model training.

**Independent Test**: Can be fully tested by configuring a training run with each split method and verifying correct data partitioning.

**Acceptance Scenarios**:
1. **Given** a dataset in a single directory, **When** the user specifies split percentages, **Then** the system partitions data into training and validation sets accordingly.
2. **Given** datasets in separate directories, **When** the user selects the directory-based split mode, **Then** the system loads training and validation sets from the specified directories.

---

### User Story 2 - Multi-Dataset Input (Priority: P2)
A user requests data from multiple datasets using the "model_inputs" configuration table, regardless of split method.

**Why this priority**: Enables advanced workflows and supports real-world use cases where data comes from multiple sources.

**Independent Test**: Can be fully tested by configuring multiple datasets in "model_inputs" and verifying correct aggregation and splitting.

**Acceptance Scenarios**:
1. **Given** multiple datasets, **When** the user configures them in "model_inputs", **Then** the system loads and splits data as specified for each dataset.

---

### User Story 3 - Minimal Configuration Overhead (Priority: P3)
A user defines their split style without duplicating configuration settings, using a clear and ergonomic interface.

**Why this priority**: Reduces user error and configuration complexity, improving usability.

**Independent Test**: Can be fully tested by reviewing configuration files for redundancy and verifying that split style is defined in a single, intuitive place.

**Acceptance Scenarios**:
1. **Given** a configuration file, **When** the user defines the split style, **Then** the system does not require duplicated or redundant settings.

## Functional Requirements

1. The system MUST support at least two split modes: "all data in one location" (with percentage/absolute split) and "data in separate directories".
2. The system MUST be designed for future extensibility, allowing additional split modes to be added with minimal changes to configuration and code.
3. The user MUST be able to select the split mode via a single, ergonomic configuration option.
4. The system MUST allow users to request data from multiple datasets using the "model_inputs" configuration table.
5. The configuration interface MUST minimize duplication and be intuitive for users.
6. The system MUST validate configuration and provide clear error messages for invalid or ambiguous split definitions.

## Success Criteria

- Users can configure either split mode and successfully train models with correct data partitioning.
- Users can aggregate data from multiple datasets and apply splits as specified.
- Configuration files are concise, with no duplicated split settings.
- All acceptance scenarios pass in end-to-end tests.
- User feedback indicates the interface is intuitive and easy to use.

## Assumptions

- Users are familiar with basic configuration files and dataset organization.
- "model_inputs" table is already available and supports multi-dataset input.
- Directory-based split mode assumes user has pre-organized data into appropriate folders.
- No additional split modes are required for initial implementation.

## Key Entities

- Dataset
- Configuration file
- Split mode (location-based, directory-based)
- model_inputs table
A user defines their split style without duplicating configuration settings, using a clear and ergonomic interface.

**Why this priority**: Reduces user error and configuration complexity, improving usability.

**Independent Test**: Can be fully tested by reviewing configuration files for redundancy and verifying that split style is defined in a single, intuitive place.

**Acceptance Scenarios**:
1. **Given** a configuration file, **When** the user defines the split style, **Then** the system does not require duplicated or redundant settings.

1. **Given** [initial state], **When** [action], **Then** [expected outcome]

---

[Add more user stories as needed, each with an assigned priority]

### Edge Cases

<!--
  ACTION REQUIRED: The content in this section represents placeholders.
  Fill them out with the right edge cases.
-->

- What happens when [boundary condition]?
- How does system handle [error scenario]?

## Requirements *(mandatory)*

<!--
  ACTION REQUIRED: The content in this section represents placeholders.
  Fill them out with the right functional requirements.
-->

### Functional Requirements

- **FR-001**: System MUST [specific capability, e.g., "allow users to create accounts"]
- **FR-002**: System MUST [specific capability, e.g., "validate email addresses"]  
- **FR-003**: Users MUST be able to [key interaction, e.g., "reset their password"]
- **FR-004**: System MUST [data requirement, e.g., "persist user preferences"]
- **FR-005**: System MUST [behavior, e.g., "log all security events"]

*Example of marking unclear requirements:*

- **FR-006**: System MUST authenticate users via [NEEDS CLARIFICATION: auth method not specified - email/password, SSO, OAuth?]
- **FR-007**: System MUST retain user data for [NEEDS CLARIFICATION: retention period not specified]

### Key Entities *(include if feature involves data)*

- **[Entity 1]**: [What it represents, key attributes without implementation]
- **[Entity 2]**: [What it represents, relationships to other entities]

## Success Criteria *(mandatory)*

<!--
  ACTION REQUIRED: Define measurable success criteria.
  These must be technology-agnostic and measurable.
-->

### Measurable Outcomes

- **SC-001**: [Measurable metric, e.g., "Users can complete account creation in under 2 minutes"]
- **SC-002**: [Measurable metric, e.g., "System handles 1000 concurrent users without degradation"]
- **SC-003**: [User satisfaction metric, e.g., "90% of users successfully complete primary task on first attempt"]
- **SC-004**: [Business metric, e.g., "Reduce support tickets related to [X] by 50%"]
