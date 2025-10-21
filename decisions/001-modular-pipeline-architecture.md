# ADR 001: Modular Pipeline Architecture

## Status

✅ **Accepted**

## Context

CytoFlow-QC processes flow cytometry data through multiple stages including data ingestion, compensation, quality control, automated gating, batch drift analysis, and statistical analysis. The system needs to handle large datasets efficiently while maintaining flexibility for different experimental designs and analysis requirements.

## Decision

We adopted a modular pipeline architecture with the following design principles:

### 1. **Stage-Based Processing**
- Each major processing step (ingest, compensate, qc, gate, drift, stats) is implemented as an independent stage
- Stages communicate through standardized file formats (Parquet for events, CSV for metadata, JSON for parameters)
- Each stage is idempotent and can be run independently for debugging and incremental processing

### 2. **Configuration-Driven Design**
- All stage behavior is controlled through YAML configuration files
- Channel mappings, QC thresholds, gating parameters, and statistical settings are externalized
- Default configurations provide sensible defaults while allowing customization

### 3. **Immutable Data Flow**
- Each stage produces new output directories rather than modifying inputs
- Original data is never modified, ensuring reproducibility
- Failed stages don't corrupt previous results

### 4. **CLI-First Interface**
- Primary interface is through a Typer-based CLI for scriptability
- Each stage can be run independently or as part of the full pipeline
- Supports both individual stage execution and end-to-end workflows

## Consequences

### Positive
- **Maintainability**: Each stage can be developed, tested, and debugged independently
- **Reproducibility**: Clear data flow and immutable processing enable exact reproduction of results
- **Flexibility**: Users can customize any stage without affecting others
- **Performance**: Stages can be optimized individually and run in parallel where appropriate
- **Debugging**: Failed stages don't affect previous work, making troubleshooting easier

### Negative
- **Disk Usage**: Multiple output directories increase storage requirements
- **I/O Overhead**: File-based communication between stages adds serialization/deserialization overhead
- **Complexity**: More moving parts require careful coordination and error handling

## Alternatives Considered

1. **Monolithic Pipeline**: Single function processing all stages - rejected due to lack of modularity
2. **In-Memory Pipeline**: Process all data in memory - rejected due to memory constraints with large datasets
3. **Database-Centric**: Store intermediate results in database - rejected due to added complexity without clear benefits

## Implementation Notes

- Each stage follows the pattern: `stage_<name>(input_dir, output_dir, config)`
- Manifest files track processing metadata and enable pipeline resumption
- Error handling ensures graceful degradation when individual samples fail
- Progress reporting provides user feedback during long-running operations

## Related ADRs

- ADR 002: File Format Selection
- ADR 003: Configuration Schema Design
- ADR 004: Error Handling Strategy










