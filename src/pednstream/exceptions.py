"""Custom Exceptions for PednStream Error Handling"""


class RequiredConfigError(Exception):
    """Configuration is required in a simulation"""


class InvalidConfigError(Exception):
    """Configuration value is invalid (e.g., OD pair references non-existent node)"""
