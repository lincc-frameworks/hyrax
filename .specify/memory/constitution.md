# Hyrax Constitution


## Core Principles

### I. Simplicity & Scientific Focus
Hyrax MUST enable astronomers to focus on science and experimentation, not boilerplate infrastructure. All features and APIs MUST prioritize ease of use and scientific clarity over unnecessary complexity.
Rationale: Lowering barriers for scientific discovery and experimentation.

### II. Reproducibility & Transparency
All results from training and inference MUST be reproducible in accordance with scientific standards. Hyrax MUST provide mechanisms for tracking configurations, environments, and random seeds, and document all steps needed to reproduce results (including environment setup, data sources, and configuration files). Published results MUST include sufficient metadata for independent verification. Documentation MUST clearly state reproducibility guarantees and limitations.
Rationale: Scientific results require reproducibility for validation and trust.

### III. Extensibility & Community Sharing
Hyrax MUST be highly extensible and encourage sharing of datasets, models, and extensions. It MUST support easy integration and provide mechanisms for extension registration and discovery (e.g., plugins, entry points, documentation patterns). Hyrax SHOULD facilitate discoverability and sharing, but is not required to maintain a central registry. Documentation MUST highlight extension points and best practices.
Rationale: Community-driven development accelerates innovation and broadens applicability.

### IV. Library-First & Interoperability
Hyrax MUST be usable as a library. Users MUST be able to freely move out of the Hyrax framework if desired. All components SHOULD rely on well-established, stable, open-source tooling, and Hyrax SHOULD provide glue code to link pipeline stages cohesively.
Rationale: Flexibility and interoperability empower users and future-proof the project.

### V. Stable Tooling & Open Source Reliance
Hyrax MUST rely on proven, stable, open-source tools wherever possible. New dependencies MUST be justified and reviewed for stability and community support. The framework SHOULD avoid reinventing wheels and instead integrate best-in-class solutions.
Rationale: Stability and sustainability are critical for long-term scientific projects.


## Scalability & Multi-Environment Support
Hyrax MUST support fluid scaling from a user's laptop to high performance computing facilities (e.g., clusters, cloud, or supercomputers) without requiring code changes. Core functionality MUST be usable from the command line, as a Slurm script, or within an interactive notebook, ensuring seamless operation across environments and interfaces.


## Governance
The Hyrax constitution supersedes all other development practices. Amendments require documentation, approval by core maintainers, and a migration plan for affected users. All pull requests and reviews MUST verify compliance with the constitution. Complexity MUST be justified in design documents. Use the developer guide and README for runtime development guidance.

**Version**: 1.1.0 | **Ratified**: TODO(RATIFICATION_DATE): original adoption date unknown | **Last Amended**: 2025-10-13
<!--
Sync Impact Report
Version change: 1.0.0 → 1.1.0
Modified principles: All replaced with Hyrax-specific principles
Added sections: Scientific Reproducibility, Extensibility & Community
Templates requiring updates: plan-template.md ✅, spec-template.md ✅, tasks-template.md ✅
Follow-up TODOs: RATIFICATION_DATE (unknown, needs confirmation)
-->