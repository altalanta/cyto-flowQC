"""Pre-flight validation checks for the CytoFlow-QC pipeline."""
from pathlib import Path
import logging

from cytoflow_qc.config import load_and_validate_config, AppConfig
from cytoflow_qc.io import load_samplesheet, get_fcs_metadata
from cytoflow_qc.exceptions import ValidationError

logger = logging.getLogger(__name__)

def validate_inputs(samplesheet_path: Path, config_path: Path) -> AppConfig:
    """
    Perform a comprehensive set of pre-flight checks on all pipeline inputs.

    Raises:
        ValidationError: if any check fails.

    Returns:
        The validated ``AppConfig`` instance.
    """
    errors: list[str] = []

    # 1. Validate config.yaml
    try:
        logger.info(f"Validating configuration file: {config_path}")
        config = load_and_validate_config(config_path)
        logger.info("✅ Configuration file is valid.")
    except Exception as e:
        errors.append(f"Configuration file validation failed: {e}")
        logger.error(f"❌ {errors[-1]}")
        raise ValidationError("; ".join(errors)) from e

    # 2. Validate samplesheet.csv
    try:
        logger.info(f"Validating samplesheet: {samplesheet_path}")
        samplesheet = load_samplesheet(str(samplesheet_path))
        if samplesheet["missing_file"].any():
            missing_files = samplesheet[samplesheet["missing_file"]]["file_path"].tolist()
            msg = "The following FCS files listed in the samplesheet are missing:"
            errors.append(msg)
            logger.error(f"❌ {msg}")
            for f in missing_files:
                logger.error(f"  - {f}")
        else:
            logger.info("✅ All FCS files listed in samplesheet exist.")
    except Exception as e:
        errors.append(f"Samplesheet validation failed: {e}")
        logger.error(f"❌ {errors[-1]}")
        raise ValidationError("; ".join(errors)) from e

    # 3. Validate channel names
    if not samplesheet.empty:
        first_fcs_path = samplesheet["file_path"].iloc[0]
        try:
            logger.info(f"Validating channel names against first FCS file: {first_fcs_path}")
            fcs_metadata = get_fcs_metadata(first_fcs_path)
            fcs_channels = {c["PnN"] for c in fcs_metadata.get("channels", [])}

            configured_channels = set(config.channels.model_dump().values())
            configured_channels.discard(None)  # Remove None if viability channel is not set

            missing_in_fcs = configured_channels - fcs_channels
            if missing_in_fcs:
                msg = "The following channels are configured in config.yaml but are missing from the FCS file:"
                errors.append(msg)
                logger.error(f"❌ {msg}")
                for ch in sorted(list(missing_in_fcs)):
                    logger.error(f"  - {ch}")
            else:
                logger.info("✅ All configured channels are present in the FCS file.")
        except Exception as e:
            errors.append(f"Failed to validate channels against FCS file: {e}")
            logger.error(f"❌ {errors[-1]}")
            raise ValidationError("; ".join(errors)) from e

    if errors:
        logger.error("\n💥 Input validation failed. Please fix the errors above before running the pipeline.")
        raise ValidationError("; ".join(errors))

    logger.info("\n🎉 All input validation checks passed successfully!")
    return config



