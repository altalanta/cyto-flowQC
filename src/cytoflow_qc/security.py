from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from cryptography.fernet import Fernet
from faker import Faker


# --- Constants ---
ENCRYPTION_KEY_ENV_VAR = "CYTOFLOW_QC_ENCRYPTION_KEY"


# --- Helper Functions ---
class SecurityError(Exception):
    """Custom exception for security-related errors."""
    pass

def _load_encryption_key() -> bytes:
    """Load encryption key from environment variable."""
    key = os.getenv(ENCRYPTION_KEY_ENV_VAR)
    if not key:
        raise SecurityError(f"Encryption key not found. Set the {ENCRYPTION_KEY_ENV_VAR} environment variable.")
    return key.encode()

def _generate_key_if_not_exists(key_path: Path | None = None) -> bytes:
    """Generate a new encryption key if one doesn't exist, or load from path.

    Args:
        key_path: Optional path to save/load the key. If None, uses environment variable.

    Returns:
        The encryption key.
    """
    if key_path and key_path.exists():
        return key_path.read_bytes()
    elif os.getenv(ENCRYPTION_KEY_ENV_VAR):
        return os.getenv(ENCRYPTION_KEY_ENV_VAR).encode()
    else:
        key = Fernet.generate_key()
        if key_path:
            key_path.write_bytes(key)
            print(f"New encryption key generated and saved to {key_path}")
        else:
            print(f"No encryption key found. Generated a new one. Store this in {ENCRYPTION_KEY_ENV_VAR}: {key.decode()}")
        return key


# --- Anonymization ---
class DataAnonymizer:
    """Handles anonymization of sensitive data in DataFrames."""

    def __init__(self, seed: int | None = None):
        """Initialize DataAnonymizer.

        Args:
            seed: Seed for Faker to ensure reproducible anonymization.
        """
        self.fake = Faker()
        if seed is not None:
            # Faker requires seeding via seed_instance for determinism
            self.fake.seed_instance(seed)
        self.mapping: dict[str, dict[str, str]] = {}

    def anonymize_dataframe(
        self, df: pd.DataFrame, columns_to_anonymize: list[str], identifier_col: str | None = None
    ) -> pd.DataFrame:
        """Anonymize specified columns in a DataFrame.

        Args:
            df: DataFrame to anonymize.
            columns_to_anonymize: List of column names to anonymize.
            identifier_col: Optional column to use as a stable identifier for mapping.

        Returns:
            Anonymized DataFrame.
        """
        df_anonymized = df.copy()

        for col in columns_to_anonymize:
            if col not in df_anonymized.columns:
                print(f"Warning: Column '{col}' not found for anonymization.")
                continue

            if identifier_col and identifier_col in df_anonymized.columns:
                # Use identifier for stable mapping
                self.mapping[col] = { original: self._anonymize_value(col, original)
                                    for original in df_anonymized[col].unique() }
                df_anonymized[col] = df_anonymized[col].map(self.mapping[col])
            else:
                # Anonymize each value directly (less stable across datasets)
                df_anonymized[col] = df_anonymized[col].apply(lambda x: self._anonymize_value(col, x))
        return df_anonymized

    def _anonymize_value(self, column: str, value: Any) -> Any:
        """Anonymize a single value based on column type or name."""
        if pd.isna(value):
            return value

        # More sophisticated anonymization logic can be added here
        if "name" in column.lower():
            return self.fake.name()
        elif "email" in column.lower():
            return self.fake.email()
        elif "address" in column.lower():
            return self.fake.address()
        elif "date" in column.lower():
            return self.fake.date_of_birth(minimum_age=18, maximum_age=90).isoformat()
        elif "patient_id" in column.lower() or "subject_id" in column.lower():
            return self.fake.uuid4()
        elif isinstance(value, (int, float)):
            # Simple numerical perturbation for demonstration
            return value + self.fake.random_int(min=-5, max=5)
        else:
            # Default to a generic anonymized string
            return self.fake.word() + "_anon"

    def get_anonymization_mapping(self) -> dict[str, dict[str, str]]:
        """Return the mapping used for anonymization.

        Returns:
            A dictionary where keys are original values and values are anonymized values.
        """
        return self.mapping


# --- Encryption / Decryption ---
class DataEncryptor:
    """Handles encryption and decryption of data."""

    def __init__(self, key: bytes | None = None, key_path: Path | None = None):
        """Initialize DataEncryptor.

        Args:
            key: Encryption key (bytes). If None, tries to load from environment or generate.
            key_path: Path to key file. Used if `key` is None.
        """
        self.key = key if key else _generate_key_if_not_exists(key_path)
        self.f = Fernet(self.key)

    def encrypt_data(self, data: bytes) -> bytes:
        """Encrypt data.

        Args:
            data: Data to encrypt (bytes).

        Returns:
            Encrypted data (bytes).
        """
        return self.f.encrypt(data)

    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt data.

        Args:
            encrypted_data: Encrypted data (bytes).

        Returns:
            Decrypted data (bytes).
        """
        return self.f.decrypt(encrypted_data)

    def encrypt_file(self, input_file: str | Path, output_file: str | Path) -> None:
        """Encrypt a file.

        Args:
            input_file: Path to input file.
            output_file: Path to output encrypted file.
        """
        input_path = Path(input_file)
        output_path = Path(output_file)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        with open(input_path, 'rb') as f_in:
            original_data = f_in.read()

        encrypted_data = self.encrypt_data(original_data)

        with open(output_path, 'wb') as f_out:
            f_out.write(encrypted_data)
        print(f"File encrypted from {input_file} to {output_file}")

    def decrypt_file(self, input_file: str | Path, output_file: str | Path) -> None:
        """Decrypt a file.

        Args:
            input_file: Path to input encrypted file.
            output_file: Path to output decrypted file.
        """
        input_path = Path(input_file)
        output_path = Path(output_file)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        with open(input_path, 'rb') as f_in:
            encrypted_data = f_in.read()

        decrypted_data = self.decrypt_data(encrypted_data)

        with open(output_path, 'wb') as f_out:
            f_out.write(decrypted_data)
        print(f"File decrypted from {input_file} to {output_file}")


# --- Role-Based Access Control (RBAC) ---
class RBACManager:
    """Manages role-based access control policies."""

    def __init__(self, policy_file: str | Path | None = None):
        """Initialize RBACManager.

        Args:
            policy_file: Path to JSON policy file.
        """
        self.policies: dict[str, Any] = self._load_policies(policy_file)

    def _load_policies(self, policy_file: str | Path | None) -> dict[str, Any]:
        """Load RBAC policies from a JSON file.

        Returns default policy if file not found.
        """
        if policy_file and Path(policy_file).exists():
            with open(policy_file, 'r') as f:
                return json.load(f)
        else:
            print("Warning: RBAC policy file not found. Using default policies.")
            return self._get_default_policies()

    def _get_default_policies(self) -> dict[str, Any]:
        """Return default RBAC policies."""
        return {
            "roles": {
                "admin": {"permissions": ["read", "write", "delete", "manage_users", "manage_policies"]},
                "analyst": {"permissions": ["read", "analyze", "export"]},
                "viewer": {"permissions": ["read", "export"]},
                "guest": {"permissions": ["read_public"]},
            },
            "resources": {
                "data_raw": {"protected_by": ["admin", "analyst"]},
                "data_processed": {"protected_by": ["admin", "analyst", "viewer"]},
                "reports": {"protected_by": ["admin", "analyst", "viewer", "guest"]},
                "configs": {"protected_by": ["admin"]},
            }
        }

    def check_permission(self, user_roles: list[str], action: str, resource: str) -> bool:
        """Check if user roles have permission to perform an action on a resource.

        Args:
            user_roles: List of roles assigned to the user.
            action: Action to perform (e.g., 'read', 'write').
            resource: Resource to access (e.g., 'data_raw', 'reports').

        Returns:
            True if permission is granted, False otherwise.
        """
        resource_policy = self.policies["resources"].get(resource)
        if not resource_policy:
            print(f"Warning: Resource '{resource}' not defined in policies.")
            return False

        allowed_roles = resource_policy.get("protected_by", [])

        for role in user_roles:
            if role in allowed_roles:
                role_permissions = self.policies["roles"].get(role, {}).get("permissions", [])
                if action in role_permissions:
                    return True
        return False

    def enforce_access(
        self, user_roles: list[str], action: str, resource: str
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator to enforce access control on functions.

        Args:
            user_roles: Roles of the user trying to access.
            action: Action being attempted.
            resource: Resource being accessed.

        Returns:
            A decorator function.
        """
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                if not self.check_permission(user_roles, action, resource):
                    raise SecurityError(f"Permission denied: User with roles {user_roles} cannot {action} {resource}")
                return func(*args, **kwargs)
            return wrapper
        return decorator


if __name__ == "__main__":
    # Example Usage

    # Anonymization
    print("--- Data Anonymization Example ---")
    data = {
        "patient_id": ["P001", "P002", "P003"],
        "patient_name": ["Alice Smith", "Bob Johnson", "Charlie Brown"],
        "email": ["alice@example.com", "bob@example.com", "charlie@example.com"],
        "age": [30, 45, 60],
        "FSC-A": [1000, 1500, 1200],
    }
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)

    anonymizer = DataAnonymizer(seed=42) # Seed for reproducibility
    columns_to_anonymize = ["patient_id", "patient_name", "email", "age"]
    df_anon = anonymizer.anonymize_dataframe(df, columns_to_anonymize, identifier_col="patient_id")

    print("\nAnonymized DataFrame:")
    print(df_anon)
    print("\nAnonymization Mapping (for 'patient_id'):")
    print(anonymizer.get_anonymization_mapping().get("patient_id"))

    # Encryption / Decryption
    print("\n--- Data Encryption/Decryption Example ---")
    encryptor = DataEncryptor(key_path=Path("encryption.key")) # Key will be generated if not exists

    test_message = b"This is a super secret message."
    encrypted_message = encryptor.encrypt_data(test_message)
    decrypted_message = encryptor.decrypt_data(encrypted_message)

    print(f"Original: {test_message}")
    print(f"Encrypted: {encrypted_message}")
    print(f"Decrypted: {decrypted_message}")

    # RBAC Example
    print("\n--- RBAC Example ---")
    rbac = RBACManager()

    # Test permissions
    admin_roles = ["admin"]
    analyst_roles = ["analyst"]
    viewer_roles = ["viewer"]
    guest_roles = ["guest"]

    print(f"Admin can read data_raw: {rbac.check_permission(admin_roles, 'read', 'data_raw')}")
    print(f"Analyst can write data_raw: {rbac.check_permission(analyst_roles, 'write', 'data_raw')}")
    print(f"Viewer can read reports: {rbac.check_permission(viewer_roles, 'read', 'reports')}")
    print(f"Guest can delete reports: {rbac.check_permission(guest_roles, 'delete', 'reports')}")

    # Example with decorator
    @rbac.enforce_access(admin_roles, "write", "configs")
    def update_config(new_config: dict[str, Any]) -> None:
        print(f"Config updated by admin: {new_config}")

    @rbac.enforce_access(analyst_roles, "analyze", "data_processed")
    def perform_analysis() -> None:
        print("Analysis performed by analyst.")

    try:
        update_config({"setting": "value"})
        perform_analysis()
    except SecurityError as e:
        print(e)

    try:
        # This should fail
        @rbac.enforce_access(viewer_roles, "write", "configs")
        def illegal_config_update() -> None:
            print("This should not be printed.")
        illegal_config_update()
    except SecurityError as e:
        print(e)






