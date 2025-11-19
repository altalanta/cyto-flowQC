"""Pre-flight validation checks for the CytoFlow-QC pipeline."""
from pathlib import Path
import logging

from cytoflow_qc.config import load_and_validate_config, AppConfig
from cytoflow_qc.io import load_samplesheet, get_fcs_metadata

logger = logging.getLogger(__name__)

def validate_inputs(samplesheet_path: Path, config_path: Path) -> bool:
    """
    Perform a comprehensive set of pre-flight checks on all pipeline inputs.

    Returns:
        True if all checks pass, False otherwise.
    """
    checks_passed = True
    
    # 1. Validate config.yaml
    try:
        logger.info(f"Validating configuration file: {config_path}")
        config = load_and_validate_config(config_path)
        logger.info("✅ Configuration file is valid.")
    except Exception as e:
        logger.error(f"❌ Configuration file validation failed: {e}")
        checks_passed = False
        return checks_passed # Stop here if config is invalid

    # 2. Validate samplesheet.csv
    try:
        logger.info(f"Validating samplesheet: {samplesheet_path}")
        samplesheet = load_samplesheet(str(samplesheet_path))
        if samplesheet["missing_file"].any():
            missing_files = samplesheet[samplesheet["missing_file"]]["file_path"].tolist()
            logger.error(f"❌ The following FCS files listed in the samplesheet are missing:")
            for f in missing_files:
                logger.error(f"  - {f}")
            checks_passed = False
        else:
            logger.info("✅ All FCS files listed in samplesheet exist.")
    except Exception as e:
        logger.error(f"❌ Samplesheet validation failed: {e}")
        checks_passed = False
        return checks_passed

    # 3. Validate channel names
    if not samplesheet.empty:
        first_fcs_path = samplesheet["file_path"].iloc[0]
        try:
            logger.info(f"Validating channel names against first FCS file: {first_fcs_path}")
            fcs_metadata = get_fcs_metadata(first_fcs_path)
            fcs_channels = {c['PnN'] for c in fcs_metadata.get('channels', [])}
            
            configured_channels = set(config.channels.dict().values())
            configured_channels.discard(None) # Remove None if viability channel is not set
            
            missing_in_fcs = configured_channels - fcs_channels
            if missing_in_fcs:
                logger.error("❌ The following channels are configured in config.yaml but are missing from the FCS file:")
                for ch in sorted(list(missing_in_fcs)):
                    logger.error(f"  - {ch}")
                checks_passed = False
            else:
                logger.info("✅ All configured channels are present in the FCS file.")

        except Exception as e:
            logger.error(f"❌ Failed to validate channels against FCS file: {e}")
            checks_passed = False
            
    if checks_passed:
        logger.info("\n🎉 All input validation checks passed successfully!")
    else:
        logger.error("\n💥 Input validation failed. Please fix the errors above before running the pipeline.")

    return checks_passed

