# Hyrax Claude Skills

This directory contains Claude Code skills for Hyrax development. Skills are reusable task patterns that provide structured guidance for common development workflows.

## Available Skills

### 1. Hyrax Development Workflow
**File**: `hyrax-development-workflow.json`

Complete guide for setup → code → test → commit workflow. Covers:
- Environment setup with conda/pip
- Validation workflow (format, test, pre-commit)
- Command timeout expectations (CRITICAL: never cancel prematurely)
- Network retry strategies for PyPI
- Best practices for development

**Use when**: Setting up dev environment, validating changes, understanding timeouts

### 2. Hyrax Testing Strategy
**File**: `hyrax-testing-strategy.json`

Comprehensive testing guide covering:
- Fast vs slow test markers (`@pytest.mark.slow`)
- Parallel execution with pytest-xdist
- Test fixtures and patterns
- Network retry for Pooch downloads
- Debugging test failures

**Use when**: Running tests, adding new tests, debugging failures

### 3. Adding Hyrax Components
**File**: `adding-hyrax-components.json`

Step-by-step guides for:
- Adding new models with `@hyrax_model` decorator
- Creating datasets with proper registration
- Implementing CLI verbs with `@hyrax_verb`
- Common registration issues and solutions

**Use when**: Adding models, datasets, or verbs to Hyrax

### 4. Hyrax Configuration System
**File**: `hyrax-configuration-system.json`

Configuration system deep dive:
- ConfigDict vs regular dict usage
- Default requirements in `hyrax_default_config.toml`
- `key = false` convention for optional features
- Configuration immutability rules
- Pydantic validation schemas

**Use when**: Working with configs, encountering "config key not found" errors

## How to Use Skills

### In Claude Code (claude.ai/code)

Skills appear in the Skills panel in Claude Code. You can:
1. Browse available skills in the Skills panel
2. Reference skills when asking Claude for help
3. Let Claude automatically suggest relevant skills

Example prompts:
- "Use the Hyrax Development Workflow skill to set up my environment"
- "Follow the Hyrax Testing Strategy to run tests"
- "Apply the Adding Hyrax Components skill to create a new model"

### Manual Reference

Skills are JSON files with structured instructions. You can:
1. Read them directly for guidance
2. Copy/paste relevant sections for reference
3. Share links to specific skills in documentation

## Skill Structure

Each skill JSON contains:
- `name`: Human-readable skill name
- `description`: Brief description of skill purpose
- `version`: Semantic version for tracking updates
- `tags`: Keywords for discovery
- `instructions`: Detailed markdown instructions

## Best Practices

1. **Consult skills proactively**: Don't wait for errors - use skills when starting tasks
2. **Follow timeouts**: Skills specify minimum timeouts for commands - NEVER cancel prematurely
3. **Reference related skills**: Skills cross-reference each other for comprehensive guidance
4. **Keep skills updated**: As Hyrax evolves, update skills to reflect current best practices

## Related Documentation

- **[CLAUDE.md](../../CLAUDE.md)**: Claude Code-specific guidance
- **[HYRAX_GUIDE.md](../../HYRAX_GUIDE.md)**: Comprehensive project documentation
- **[copilot-instructions.md](../copilot-instructions.md)**: GitHub Copilot guidance

## Contributing

When updating skills:
1. Maintain JSON structure (name, description, version, tags, instructions)
2. Use clear, actionable markdown in instructions
3. Include code examples and command reference
4. Cross-reference related skills
5. Update version numbers semantically
6. Validate JSON syntax after changes
