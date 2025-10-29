# Data Security and Privacy

CytoFlow-QC now includes enhanced data security and privacy features to help users manage sensitive biological and clinical data. These features include data anonymization, encryption/decryption, and a basic role-based access control (RBAC) system.

## Data Anonymization

The data anonymization features allow you to remove or mask sensitive identifiers from your flow cytometry data, ensuring patient privacy and compliance with regulations like HIPAA and GDPR.

### CLI Usage

Use the `cytoflow-qc anonymize` command to anonymize specified columns in your dataframes:

```bash
cytoflow-qc anonymize \
    /path/to/input/dataframes \
    /path/to/output/anonymized_data \
    --columns patient_id,patient_name,email \
    --identifier patient_id
```

**Arguments:**

*   `<input-directory>`: Path to the directory containing your input dataframes (e.g., `.parquet` files).
*   `<output-directory>`: Path to the directory where anonymized dataframes will be saved.
*   `--columns`, `-c`: A comma-separated list of column names to anonymize (e.g., `patient_id,patient_name,email`).
*   `--identifier`, `-i`: (Optional) A column to use as a stable identifier for consistent anonymization across multiple datasets. If provided, the same original value in this column will always map to the same anonymized value.

### Programmatic Usage

You can also use the `DataAnonymizer` class directly in your Python scripts:

```python
from cytoflow_qc.security import DataAnonymizer
import pandas as pd

data = {
    "patient_id": ["P001", "P002", "P003"],
    "patient_name": ["Alice Smith", "Bob Johnson", "Charlie Brown"],
    "email": ["alice@example.com", "bob@example.com", "charlie@example.com"],
    "age": [30, 45, 60],
    "FSC-A": [1000, 1500, 1200],
}
df = pd.DataFrame(data)

anonymizer = DataAnonymizer(seed=42) # Use a seed for reproducible anonymization
columns_to_anonymize = ["patient_id", "patient_name", "email", "age"]
df_anon = anonymizer.anonymize_dataframe(df, columns_to_anonymize, identifier_col="patient_id")

print("Original DataFrame:")
print(df)
print("\nAnonymized DataFrame:")
print(df_anon)
print("\nAnonymization Mapping (for 'patient_id'):")
print(anonymizer.get_anonymization_mapping().get("patient_id"))
```

## Data Encryption and Decryption

To protect sensitive data at rest and in transit, CytoFlow-QC provides tools for symmetric encryption and decryption of files.

### CLI Usage

**Encrypt a file:**

```bash
cytoflow-qc encrypt \
    /path/to/my_sensitive_data.csv \
    /path/to/my_sensitive_data.encrypted \
    --key-path /path/to/encryption.key
```

**Decrypt a file:**

```bash
cytoflow-qc decrypt \
    /path/to/my_sensitive_data.encrypted \
    /path/to/my_sensitive_data.decrypted.csv \
    --key-path /path/to/encryption.key
```

**Arguments:**

*   `<input-file>`: Path to the file to be encrypted or decrypted.
*   `<output-file>`: Path where the encrypted or decrypted file will be saved.
*   `--key-path`, `-k`: (Optional) Path to a file containing the encryption key. If not provided and no `CYTOFLOW_QC_ENCRYPTION_KEY` environment variable is set, a new key will be generated and printed to the console (you should save this securely).

### Programmatic Usage

You can use the `DataEncryptor` class in your Python scripts:

```python
from cytoflow_qc.security import DataEncryptor
from pathlib import Path
import os

# Generate a key if not available (or load from file/env var)
# If key_path is not provided and env var not set, it will print the key to console.
encryptor = DataEncryptor(key_path=Path("encryption.key")) 

# Encrypt data
sensitive_bytes = b"This is some highly confidential data."
encrypted_bytes = encryptor.encrypt_data(sensitive_bytes)
print(f"Encrypted: {encrypted_bytes}")

# Decrypt data
decrypted_bytes = encryptor.decrypt_data(encrypted_bytes)
print(f"Decrypted: {decrypted_bytes}")

# Encrypt/Decrypt files
# Assuming 'raw_data.csv' exists
# encryptor.encrypt_file("raw_data.csv", "raw_data.encrypted")
# encryptor.decrypt_file("raw_data.encrypted", "raw_data.decrypted.csv")
```

## Role-Based Access Control (RBAC)

The RBAC system allows you to define roles and permissions to restrict access to specific resources (e.g., raw data, processed data, reports, configuration files). This ensures that only authorized users can perform certain actions.

### CLI Usage

Use the `cytoflow-qc rbac` command to check permissions:

```bash
cytoflow-qc rbac \
    read \
    data_raw \
    --roles admin,analyst
```

**Arguments:**

*   `<action>`: The action to check permission for (e.g., `read`, `write`, `delete`, `analyze`, `export`).
*   `<resource>`: The resource to access (e.g., `data_raw`, `data_processed`, `reports`, `configs`).
*   `--roles`, `-r`: A comma-separated list of roles assigned to the user (e.g., `admin`, `analyst`, `viewer`).
*   `--policy-file`, `-p`: (Optional) Path to a custom JSON policy file. If not provided, default policies will be used.

### Default Policies

Default roles and their permissions:

*   **admin**: `read`, `write`, `delete`, `manage_users`, `manage_policies`
*   **analyst**: `read`, `analyze`, `export`
*   **viewer**: `read`, `export`
*   **guest**: `read_public`

Default resources and the roles that can access them:

*   **data_raw**: `admin`, `analyst`
*   **data_processed**: `admin`, `analyst`, `viewer`
*   **reports**: `admin`, `analyst`, `viewer`, `guest`
*   **configs**: `admin`

### Custom Policy File Example

You can define your own `policy.json` file:

```json
{
    "roles": {
        "custom_admin": {"permissions": ["read", "write", "custom_action"]},
        "custom_viewer": {"permissions": ["read"]}
    },
    "resources": {
        "my_custom_data": {"protected_by": ["custom_admin"]},
        "public_reports": {"protected_by": ["custom_admin", "custom_viewer"]}
    }
}
```

### Programmatic Usage

You can use the `RBACManager` class and its decorator for programmatic access control:

```python
from cytoflow_qc.security import RBACManager, SecurityError

rbac = RBACManager(policy_file="my_custom_policy.json") # Load custom policies

user_roles = ["analyst"]

# Check permission directly
if rbac.check_permission(user_roles, "analyze", "data_processed"):
    print("Analyst can analyze processed data.")

# Use decorator to enforce access
@rbac.enforce_access(["admin"], "write", "configs")
def update_application_config(new_config: dict) -> None:
    print(f"Config updated by admin: {new_config}")

@rbac.enforce_access(user_roles, "analyze", "data_processed")
def perform_sensitive_analysis() -> None:
    print("Sensitive analysis performed by analyst.")

try:
    update_application_config({"setting": "new_value"})
    perform_sensitive_analysis()
    # This should fail if the viewer role tries to write to configs
    @rbac.enforce_access(["viewer"], "write", "configs")
    def illegal_action() -> None:
        print("This should not be printed.")
    illegal_action()
except SecurityError as e:
    print(f"Access denied: {e}")
```







